
import time
import numpy as np
import netCDF4 as nc
from scipy import spatial
from scipy.sparse import csr_matrix
import argparse

# Authors: Darren Engwirda


def map_to_r3(mesh, xlon, ylat, head, tail):
    """
    Map lon-lat coordinates to XYZ points. Restricted to the
    panel LAT[HEAD:TAIL] to manage memory use.

    """

    sinx = np.sin(xlon * np.pi / 180.)
    cosx = np.cos(xlon * np.pi / 180.)
    siny = np.sin(ylat * np.pi / 180.)
    cosy = np.cos(ylat * np.pi / 180.)

    sinu, sinv = np.meshgrid(sinx, siny[head:tail])
    cosu, cosv = np.meshgrid(cosx, cosy[head:tail])

    rsph = mesh.sphere_radius

    xpos = rsph * cosu * cosv
    ypos = rsph * sinu * cosv
    zpos = rsph * sinv

    return np.vstack(
        (xpos.ravel(), ypos.ravel(), zpos.ravel())).T


def tria_area(rs, pa, pb, pc):
    """
    Calculate areas of spherical triangles [PA, PB, PC] on a
    sphere of radius RS.

    """

    lena = circ_dist(1., pa, pb)
    lenb = circ_dist(1., pb, pc)
    lenc = circ_dist(1., pc, pa)

    slen = 0.5 * (lena + lenb + lenc)

    tana = np.tan(0.5 * (slen - lena))
    tanb = np.tan(0.5 * (slen - lenb))
    tanc = np.tan(0.5 * (slen - lenc))

    edel = 4.0 * np.arctan(np.sqrt(
        np.tan(0.5 * slen) * tana * tanb * tanc))

    return edel * rs ** 2


def circ_dist(rs, pa, pb):
    """
    Calculate geodesic-length of great circles [PA, PB] on a
    sphere of radius RS.

    """

    dlon = .5 * (pa[:, 0] - pb[:, 0])
    dlat = .5 * (pa[:, 1] - pb[:, 1])

    dist = 2. * rs * np.arcsin(np.sqrt(
        np.sin(dlat) ** 2 +
        np.sin(dlon) ** 2 * np.cos(pa[:, 1]) * np.cos(pb[:, 1])
    ))

    return dist


def find_vals(xlon, ylat, vals, xpos, ypos):

    cols = xlon.size - 1
    rows = ylat.size - 1

    dlon = (xlon[-1] - xlon[+0]) / cols
    dlat = (ylat[-1] - ylat[+0]) / rows

    icol = (xpos - np.min(xlon)) / dlon
    irow = (ypos - np.min(ylat)) / dlat

    icol = np.asarray(icol, dtype=np.uint32)
    irow = np.asarray(irow, dtype=np.uint32)

    return vals[irow, icol]


def cell_quad(mesh, xlon, ylat, vals):
    """
    Eval. finite-volume integrals for mesh cells - splitting
    each into a series of triangles and employing an order-2
    quadrature rule.

    """

    class base:
        pass

    abar = np.zeros(
        (mesh.dimensions["nCells"].size, 1), dtype=np.float32)
    fbar = np.zeros(
        (mesh.dimensions["nCells"].size, 1), dtype=np.float32)

    pcel = np.zeros(
        (mesh.dimensions["nCells"].size, 2), dtype=np.float64)
    pcel[:, 0] = mesh.variables["lonCell"][:]
    pcel[:, 1] = mesh.variables["latCell"][:]

    pcel = pcel * 180. / np.pi
    pcel[pcel[:, 0] > 180., 0] -= 360.

    fcel = find_vals(xlon, ylat, vals, pcel[:, 0], pcel[:, 1])

    pcel = pcel * np.pi / 180.

    pvrt = np.zeros(
        (mesh.dimensions["nVertices"].size, 2),
        dtype=np.float64)
    pvrt[:, 0] = mesh.variables["lonVertex"][:]
    pvrt[:, 1] = mesh.variables["latVertex"][:]

    pvrt = pvrt * 180. / np.pi
    pvrt[pvrt[:, 0] > 180., 0] -= 360.

    fvrt = find_vals(xlon, ylat, vals, pvrt[:, 0], pvrt[:, 1])

    pvrt = pvrt * np.pi / 180.

    cell = base()
    cell.edge = np.asarray(
        mesh.variables["edgesOnCell"][:], dtype=np.int32)
    cell.topo = np.asarray(
        mesh.variables["nEdgesOnCell"][:], dtype=np.int32)

    edge = base()
    edge.vert = np.asarray(
        mesh.variables["verticesOnEdge"][:], dtype=np.int32)

    for epos in range(np.max(cell.topo)):

        mask = cell.topo > epos

        icel = np.argwhere(mask).ravel()

        ifac = cell.edge[mask, epos] - 1

        ivrt = edge.vert[ifac, 0] - 1
        jvrt = edge.vert[ifac, 1] - 1

        rsph = mesh.sphere_radius

        atri = tria_area(
            rsph, pcel[icel], pvrt[ivrt], pvrt[jvrt])

        atri = np.reshape(atri, (atri.size, 1))

        ftri = (fcel[icel] + fvrt[ivrt] + fvrt[jvrt])

        ftri = np.reshape(ftri, (ftri.size, 1))

        abar[icel] += atri
        fbar[icel] += atri * ftri / 3.

    return fbar / abar


def dem_remap(elev_file, mpas_file):
    """
    Map elevation and ice+ocn-thickness data from a "zipped"
    RTopo data-set onto the cells in an MPAS mesh.

    Cell values are a blending of an approx. integral remap
    and a local quadrature rule.

    """

    print("Loading assests...")

    elev = nc.Dataset(elev_file, "r")
    mesh = nc.Dataset(mpas_file, "r+")

# -- Compute an approximate remapping, associating pixels in
# -- the DEM with cells in the MPAS mesh. Since polygons are
# -- Voronoi, the point-in-cell query can be computed by
# -- finding nearest neighbours. This remapping is an approx.
# -- as no partial pixel-cell intersection is computed.

    print("Building KDtree...")

    ppos = np.zeros(
        (mesh.dimensions["nCells"].size, 3), dtype=np.float64)
    ppos[:, 0] = mesh["xCell"][:]
    ppos[:, 1] = mesh["yCell"][:]
    ppos[:, 2] = mesh["zCell"][:]

    tree = spatial.cKDTree(ppos, leafsize=8)

    print("Remap elevation...")

    xlon = np.asarray(elev["lon"][:], dtype=np.float64)
    ylat = np.asarray(elev["lat"][:], dtype=np.float64)

    xmid = .5 * (xlon[:-1:] + xlon[1::])
    ymid = .5 * (ylat[:-1:] + ylat[1::])

    indx = np.asarray(np.round(
        np.linspace(-1, ymid.size, 9)), dtype=np.int32)

    print("* process tiles:")

    nset = []
    for tile in range(indx.size - 1):

        head = indx[tile + 0] + 1
        tail = indx[tile + 1] + 1

        qpos = map_to_r3(mesh, xmid, ymid, head, tail)

        ttic = time.time()
        __, nloc = tree.query(qpos, workers=-1)
        ttoc = time.time()
        print("* built node-to-cell map:", ttoc - ttic)

        nset.append(nloc)

    near = np.concatenate(nset)

    del tree
    del qpos
    del nset
    del nloc

# -- Build cell-to-pixel map as a sparse matrix, and average
# -- RTopo pixel values within each cell.

    cols = np.arange(0, near.size)
    vals = np.ones(near.size, dtype=np.int8)

    smat = csr_matrix((vals, (near, cols)))

    nmap = np.asarray(
        np.sum(smat, axis=1), dtype=np.float32)

    vals = np.asarray(
        elev["bed_elevation"][:], dtype=np.float32)
    vals = np.reshape(vals, (vals.size, 1))

    emap = (smat * vals) / np.maximum(1., nmap)

    vals = np.asarray(
        elev["ocn_thickness"][:], dtype=np.float32)
    vals = np.reshape(vals, (vals.size, 1))

    omap = (smat * vals) / np.maximum(1., nmap)

    vals = np.asarray(
        elev["ice_thickness"][:], dtype=np.float32)
    vals = np.reshape(vals, (vals.size, 1))

    imap = (smat * vals) / np.maximum(1., nmap)

    del smat
    del cols
    del vals
    del near

# -- If the resolution of the mesh is greater, or comparable
# -- to that of the DEM, the approx. remapping (above) will
# -- result in a low order interpolation.
# -- Thus, blend with a local order-2 quadrature formulation

    print("Eval. elevation...")

    vals = np.asarray(
        elev["bed_elevation"][:], dtype=np.float32)

    ttic = time.time()
    eint = cell_quad(mesh, xlon, ylat, vals)
    ttoc = time.time()
    print("* compute cell integrals:", ttoc - ttic)

    vals = np.asarray(
        elev["ocn_thickness"][:], dtype=np.float32)

    ttic = time.time()
    oint = cell_quad(mesh, xlon, ylat, vals)
    ttoc = time.time()
    print("* compute cell integrals:", ttoc - ttic)

    vals = np.asarray(
        elev["ice_thickness"][:], dtype=np.float32)

    ttic = time.time()
    iint = cell_quad(mesh, xlon, ylat, vals)
    ttoc = time.time()
    print("* compute cell integrals:", ttoc - ttic)

    print("Save to dataset...")

    ebar = (np.multiply(nmap, emap) + 6 * eint) / (6 + nmap)
    obar = (np.multiply(nmap, omap) + 6 * oint) / (6 + nmap)
    ibar = (np.multiply(nmap, imap) + 6 * iint) / (6 + nmap)

    if ("bed_elevation" not in mesh.variables.keys()):
        mesh.createVariable("bed_elevation", "f8", ("nCells"))

    if ("ocn_thickness" not in mesh.variables.keys()):
        mesh.createVariable("ocn_thickness", "f8", ("nCells"))

    if ("ice_thickness" not in mesh.variables.keys()):
        mesh.createVariable("ice_thickness", "f8", ("nCells"))

    mesh["bed_elevation"][:] = ebar
    mesh["ocn_thickness"][:] = obar
    mesh["ice_thickness"][:] = ibar

    elev.close()
    mesh.close()


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--mpas-file", dest="mpas_file", type=str,
        required=True, help="Name of user MPAS mesh.")

    parser.add_argument(
        "--elev-file", dest="elev_file", type=str,
        required=True, help="Name of DEM pixel file.")

    args = parser.parse_args()
    elev_file = args.elev_file
    mpas_file = args.mpas_file
    dem_remap(elev_file, mpas_file)
