
import argparse
import time

import netCDF4 as nc
import numpy as np
from scipy import spatial
from scipy.sparse import csr_matrix

from compass.ocean.tests.tides.dem.dem_names import names

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


def tria_area(pa, pb, pc, rs=1.):
    """
    Calculate areas of spherical triangles [PA, PB, PC] on a
    sphere of radius RS.

    """

    lena = circ_dist(
        pa[:, 0], pa[:, 1], pb[:, 0], pb[:, 1], 1.)
    lenb = circ_dist(
        pb[:, 0], pb[:, 1], pc[:, 0], pc[:, 1], 1.)
    lenc = circ_dist(
        pc[:, 0], pc[:, 1], pa[:, 0], pa[:, 1], 1.)

    slen = 0.5 * (lena + lenb + lenc)

    tana = np.tan(0.5 * (slen - lena))
    tanb = np.tan(0.5 * (slen - lenb))
    tanc = np.tan(0.5 * (slen - lenc))

    edel = 4.0 * np.arctan(np.sqrt(
        np.tan(0.5 * slen) * tana * tanb * tanc))

    return edel * rs ** 2


def circ_dist(xa, ya, xb, yb, rs=1.):
    """
    Calculate geodesic-length of great circles [PA, PB] on a
    sphere of radius RS.

    """

    dlon = .5 * (xa - xb)
    dlat = .5 * (ya - yb)

    dist = 2. * rs * np.arcsin(np.sqrt(
        np.sin(dlat) ** 2 +
        np.sin(dlon) ** 2 * np.cos(ya) * np.cos(yb)
    ))

    return dist


def sample_1d(xpos, ffun, xnew):
    """
    A fast(er) 1-dim. interpolation routine for DEM sampling.

    """

    ipos = np.searchsorted(xpos, xnew, side="left")

    ipos = np.maximum(ipos, 1)
    ipos = np.minimum(ipos, xpos.size - 1)

    spac = np.diff(xpos)
    scal = (xnew - xpos[ipos - 1]) / spac[ipos - 1]

    fnew = (
        (1.0 - scal) * ffun[ipos - 1] +
        (0.0 + scal) * ffun[ipos - 0]
    )

    return fnew


def linear_2d(xxww, yyss, xxee, yynn,
              xpos, ypos,
              ffsw, ffse, ffne, ffnw):
    """
    Standard bilinear interpolation on aligned quadrilateral.

    """

    aane = (xxee - xpos) * (yynn - ypos)
    aanw = (xpos - xxww) * (yynn - ypos)
    aase = (xxee - xpos) * (ypos - yyss)
    aasw = (xpos - xxww) * (ypos - yyss)

    asum = (aane + aanw + aase + aasw)

    return (ffnw * aase + ffne * aasw +
            ffsw * aane + ffse * aanw) / asum


def sample_2d(xlon, ylat, vals, xpos, ypos):
    """
    A fast(er) 2-dim. interpolation routine for DEM sampling.

    """

    cols = xlon.size - 1
    rows = ylat.size - 1

    dlon = (xlon[-1] - xlon[+0]) / cols
    dlat = (ylat[-1] - ylat[+0]) / rows

    icol = (xpos - np.min(xlon)) / dlon
    irow = (ypos - np.min(ylat)) / dlat

    icol = np.asarray(icol, dtype=int)
    irow = np.asarray(irow, dtype=int)

    xmid = .5 * (xlon[:-1:] + xlon[1::])
    ymid = .5 * (ylat[:-1:] + ylat[1::])

    cols = cols - 1
    rows = rows - 1
    zero = 0
    wcol = icol - 1
    wcol[wcol < zero] = cols
    ecol = icol + 1
    ecol[ecol > cols] = zero
    nrow = irow + 1
    nrow[nrow > rows] = rows
    srow = irow - 1
    srow[srow < zero] = zero

    # -- Sub-pixel bilinear interpolation - pixel DEM values are
    # -- considered to be located at pixel centres; edge middle
    # -- and corner values are reconstructed using simple linear
    # -- averaging. Each pixel is thus split into 4 "sub-pixels"
    # -- with standard bilinear interpolation applied for each.

    # -- Sub-pixel scheme preserves min/max DEM values - without
    # -- exessive smoothing.

    vmid = vals[irow, icol]

    vvnw = 0.25 * (
        vals[irow, icol] + vals[irow, wcol] +
        vals[nrow, icol] + vals[nrow, wcol]
    )
    vvne = 0.25 * (
        vals[irow, icol] + vals[irow, ecol] +
        vals[nrow, icol] + vals[nrow, ecol]
    )
    vvsw = 0.25 * (
        vals[irow, icol] + vals[irow, wcol] +
        vals[srow, icol] + vals[srow, wcol]
    )
    vvse = 0.25 * (
        vals[irow, icol] + vals[irow, ecol] +
        vals[srow, icol] + vals[srow, ecol]
    )

    vvnn = 0.50 * (
        vals[irow, icol] + vals[nrow, icol]
    )
    vvee = 0.50 * (
        vals[irow, icol] + vals[irow, ecol]
    )
    vvss = 0.50 * (
        vals[irow, icol] + vals[srow, icol]
    )
    vvww = 0.50 * (
        vals[irow, icol] + vals[irow, wcol]
    )

    isnw = np.logical_and(
        xpos <= xmid[icol + 0], ypos >= ymid[irow + 0]
    )
    isne = np.logical_and(
        xpos >= xmid[icol + 0], ypos >= ymid[irow + 0]
    )
    issw = np.logical_and(
        xpos <= xmid[icol + 0], ypos <= ymid[irow + 0]
    )
    isse = np.logical_and(
        xpos >= xmid[icol + 0], ypos <= ymid[irow + 0]
    )

    vnew = np.zeros(vmid.shape, dtype=vals.dtype)

    vnew[isnw] = linear_2d(
        xlon[icol[isnw] + 0], ymid[irow[isnw] + 0],
        xmid[icol[isnw] + 0], ylat[irow[isnw] + 1],
        xpos[isnw], ypos[isnw],
        vvww[isnw], vmid[isnw], vvnn[isnw], vvnw[isnw]
    )

    vnew[isne] = linear_2d(
        xmid[icol[isne] + 0], ymid[irow[isne] + 0],
        xlon[icol[isne] + 1], ylat[irow[isne] + 1],
        xpos[isne], ypos[isne],
        vmid[isne], vvee[isne], vvne[isne], vvnn[isne]
    )

    vnew[issw] = linear_2d(
        xlon[icol[issw] + 0], ylat[irow[issw] + 0],
        xmid[icol[issw] + 0], ymid[irow[issw] + 0],
        xpos[issw], ypos[issw],
        vvsw[issw], vvss[issw], vmid[issw], vvww[issw]
    )

    vnew[isse] = linear_2d(
        xmid[icol[isse] + 0], ylat[irow[isse] + 0],
        xlon[icol[isse] + 1], ymid[irow[isse] + 0],
        xpos[isse], ypos[isse],
        vvss[isse], vvse[isse], vvee[isse], vmid[isse]
    )

    return vnew


def cell_quad(mesh, xlon, ylat, vals):
    """
    Eval. finite-volume integrals for mesh cells - splitting
    each into a series of triangles and employing an order-2
    quadrature rule.

    """

    class base:
        pass

    ncel = mesh.dimensions["nCells"].size
    pcel = np.zeros((ncel, 2), dtype=np.float64)
    pcel[:, 0] = mesh.variables["lonCell"][:]
    pcel[:, 1] = mesh.variables["latCell"][:]

    pcel = pcel * 180. / np.pi
    pcel[pcel[:, 0] > 180., 0] -= 360.

    fcel = sample_2d(xlon, ylat, vals, pcel[:, 0], pcel[:, 1])

    pcel = pcel * np.pi / 180.

    nvrt = mesh.dimensions["nVertices"].size
    pvrt = np.zeros((nvrt, 2), dtype=np.float64)
    pvrt[:, 0] = mesh.variables["lonVertex"][:]
    pvrt[:, 1] = mesh.variables["latVertex"][:]

    pvrt = pvrt * 180. / np.pi
    pvrt[pvrt[:, 0] > 180., 0] -= 360.

    fvrt = sample_2d(xlon, ylat, vals, pvrt[:, 0], pvrt[:, 1])

    pvrt = pvrt * np.pi / 180.

    cell = base()
    cell.edge = mesh.variables["edgesOnCell"][:, :]
    cell.topo = mesh.variables["nEdgesOnCell"][:]

    edge = base()
    edge.vert = mesh.variables["verticesOnEdge"][:]

    abar = np.zeros((ncel, 1), dtype=np.float64)
    fbar = np.zeros((ncel, 1), dtype=np.float64)

    for epos in range(np.max(cell.topo)):

        mask = cell.topo > epos

        icel = np.argwhere(mask).ravel()

        ifac = cell.edge[mask, epos] - 1

        ivrt = edge.vert[ifac, 0] - 1
        jvrt = edge.vert[ifac, 1] - 1

        rsph = mesh.sphere_radius

        atri = tria_area(
            pcel[icel], pvrt[ivrt], pvrt[jvrt], rsph)

        atri = np.reshape(atri, (atri.size, 1))

        ftri = (fcel[icel] + fvrt[ivrt] + fvrt[jvrt])

        ftri = np.reshape(ftri, (ftri.size, 1))

        abar[icel] += atri
        fbar[icel] += atri * ftri / 3.

    fbar = np.asarray(fbar / abar, dtype=np.float32)

    return fvrt, fcel, fbar


def cell_prfl(mesh, smat,
              nlev, zdem, zvrt, zcel, zbar):
    """
    Build elev. profiles for each cell in the mesh - sorting
    the DEM pixel values assigned to each cell and assigning
    to NLEV bins.

    """

    prfl = np.tile(zbar, (1, nlev))

    nvrt = mesh.variables["nEdgesOnCell"][:]
    ivrt = mesh.variables["verticesOnCell"][:, :] - 1

    for cell in range(0, smat.shape[0]):

        # -- extract set of DEM pixels per MPAS cell. Sparse SMAT
        # -- contains cell-to-DEM mapping: ith row is the list of
        # -- pixels overlapping with the ith cell

        head = smat.indptr[cell + 0] + 0
        tail = smat.indptr[cell + 1] + 0
        idem = smat.indices[head:tail]

        # -- build the cell elev. profiles: sort pixels by height
        # -- and (linearly) interpolate to profile band endpoints

        if (idem.size > nvrt[cell] + 1):

            # -- list of contained DEM pixel values, for smooth cases
            nvec = idem.shape[+0]
            zvec = zdem[idem]

            prfl[cell, :] = np.interp(
                np.linspace(0., nvec - 1., nlev),
                np.arange(0., nvec),
                np.sort(zvec))

        else:

            # -- cell vert. + centre interpolations, for coarse cases
            nvec = nvrt[cell] + 1
            zvec = zvrt[ivrt[cell, :nvrt[cell]]]

            prfl[cell, :] = np.interp(
                np.linspace(0., nvec - 1., nlev),
                np.arange(0., nvec),
                np.sort(np.append(zvec, zcel[cell])))

    return prfl


def dem_remap(elev_file, mpas_file, elev_band=0):  # noqa: C901
    """
    Map elevation and ice+ocn-thickness data from a "zipped"
    RTopo data-set onto the cells in an MPAS mesh.

    Cell values are a blending of an approx. integral remap
    and a local quadrature rule.

    """

    NLEV = elev_band + 1   # no. of evel. profile bands

    print("Loading assests...")

    elev = nc.Dataset(elev_file, "r+")
    mesh = nc.Dataset(mpas_file, "r+")

    xlon = np.asarray(elev["lon"][:], dtype=np.float64)
    ylat = np.asarray(elev["lat"][:], dtype=np.float64)

    # -- add dummy data to elev file if missing

    if ("bed_elevation" not in elev.variables.keys()):
        print("*bed_elevation variable not found")
        elev.createVariable("bed_elevation",
                            "i2", ("num_row", "num_col"))

    if ("bed_slope" not in elev.variables.keys()):
        print("*bed_slope variable not found")
        elev.createVariable("bed_slope",
                            "f4", ("num_row", "num_col"))

    if ("bed_dz_dx" not in elev.variables.keys()):
        print("*bed_dz_dx variable not found")
        elev.createVariable("bed_dz_dx",
                            "f4", ("num_row", "num_col"))

    if ("bed_dz_dy" not in elev.variables.keys()):
        print("*bed_dz_dy variable not found")
        elev.createVariable("bed_dz_dy",
                            "f4", ("num_row", "num_col"))

    if ("ocn_thickness" not in elev.variables.keys()):
        print("*ocn_thickness variable not found")
        elev.createVariable("ocn_thickness",
                            "i2", ("num_row", "num_col"))

    if ("ice_thickness" not in elev.variables.keys()):
        print("*ice_thickness variable not found")
        elev.createVariable("ice_thickness",
                            "i2", ("num_row", "num_col"))

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

    tree = spatial.cKDTree(ppos, leafsize=32)

    print("Remap elevation...")

    xmid = .5 * (xlon[:-1:] + xlon[1::])
    ymid = .5 * (ylat[:-1:] + ylat[1::])

    indx = np.asarray(np.round(
        np.linspace(-1, ymid.size, 32)), dtype=int)

    print("* process tiles:")

    nset = []
    for tile in range(indx.size - 1):

        head = indx[tile + 0] + 1
        tail = indx[tile + 1] + 1

        qpos = map_to_r3(mesh, xmid, ymid, head, tail)

        ttic = time.time()
        try:  # ridiculous argument renaming...
            __, cell = tree.query(qpos, n_jobs=-1)
        except:  # noqa: E722
            __, cell = tree.query(qpos, workers=-1)
        ttoc = time.time()
        print("* built node-to-cell map:",
              tile, "of", indx.size - 1)

        nset.append(
            np.asarray(cell, dtype=np.uint32))

    del tree
    del ppos
    del qpos

    near = np.concatenate(nset)

    del nset
    del cell

    # -- Build cell-to-pixel map as a sparse matrix, and average
    # -- RTopo pixel values within each cell.

    print("Form map matrix...")

    ttic = time.time()

    ncel = mesh.dimensions["nCells"].size

    cols = np.arange(0, near.size, dtype=np.uint32)
    vals = np.ones(near.size, dtype=np.int8)

    smat = csr_matrix((vals, (near, cols)),
                      shape=(ncel, near.size), dtype=np.int8)

    del near
    del cols
    del vals

    nmap = np.asarray(
        smat.sum(axis=1), dtype=np.float32)

    vals = np.asarray(
        elev["bed_elevation"][:], dtype=np.float32)
    vals = np.reshape(vals, (vals.size, 1))

    emap = (smat * vals) / np.maximum(1., nmap)

    vals = np.asarray(
        elev["bed_slope"][:], dtype=np.float32)
    vals = np.reshape(vals, (vals.size, 1))

    smap = (smat * vals) / np.maximum(1., nmap)

    vals = np.asarray(
        elev["bed_dz_dx"][:], dtype=np.float32)
    vals = np.reshape(vals, (vals.size, 1))

    xmap = (smat * vals) / np.maximum(1., nmap)

    vals = np.asarray(
        elev["bed_dz_dy"][:], dtype=np.float32)
    vals = np.reshape(vals, (vals.size, 1))

    ymap = (smat * vals) / np.maximum(1., nmap)

    vals = np.asarray(
        elev["ocn_thickness"][:], dtype=np.float32)
    vals = np.reshape(vals, (vals.size, 1))

    omap = (smat * vals) / np.maximum(1., nmap)

    frac = np.zeros(vals.shape, dtype=vals.dtype)
    frac[vals > 0.0] = 1.
    ofrc = (smat * frac) / np.maximum(1., nmap)

    vals = np.asarray(
        elev["ice_thickness"][:], dtype=np.float32)
    vals = np.reshape(vals, (vals.size, 1))

    imap = (smat * vals) / np.maximum(1., nmap)

    frac = np.zeros(vals.shape, dtype=vals.dtype)
    frac[vals > 0.0] = 1.
    ifrc = (smat * frac) / np.maximum(1., nmap)

    ttoc = time.time()

    del vals
    del frac

    print("* built remapping matrix:",
          np.round(ttoc - ttic, decimals=1), "sec")

    # -- If the resolution of the mesh is greater, or comparable
    # -- to that of the DEM, the approx. remapping (above) will
    # -- result in a low order interpolation.
    # -- Thus, blend with a local order-2 quadrature formulation

    print("Eval. elevation...")

    vals = np.asarray(
        elev["bed_elevation"][:], dtype=np.float32)

    ttic = time.time()
    evrt, ecel, eint = cell_quad(mesh, xlon, ylat, vals)
    ttoc = time.time()
    print("* compute cell integrals:",
          np.round(ttoc - ttic, decimals=1), "sec")

    vals = np.asarray(
        elev["bed_slope"][:], dtype=np.float32)

    ttic = time.time()
    svrt, scel, sint = cell_quad(mesh, xlon, ylat, vals)
    ttoc = time.time()
    print("* compute cell integrals:",
          np.round(ttoc - ttic, decimals=1), "sec")

    vals = np.asarray(
        elev["bed_dz_dx"][:], dtype=np.float32)

    ttic = time.time()
    xvrt, xcel, xint = cell_quad(mesh, xlon, ylat, vals)
    ttoc = time.time()
    print("* compute cell integrals:",
          np.round(ttoc - ttic, decimals=1), "sec")

    vals = np.asarray(
        elev["bed_dz_dy"][:], dtype=np.float32)

    ttic = time.time()
    yvrt, ycel, yint = cell_quad(mesh, xlon, ylat, vals)
    ttoc = time.time()
    print("* compute cell integrals:",
          np.round(ttoc - ttic, decimals=1), "sec")

    vals = np.asarray(
        elev["ocn_thickness"][:], dtype=np.float32)

    ttic = time.time()
    ovrt, ocel, oint = cell_quad(mesh, xlon, ylat, vals)
    ttoc = time.time()
    print("* compute cell integrals:",
          np.round(ttoc - ttic, decimals=1), "sec")

    vals = np.asarray(
        elev["ice_thickness"][:], dtype=np.float32)

    ttic = time.time()
    ivrt, icel, iint = cell_quad(mesh, xlon, ylat, vals)
    ttoc = time.time()
    print("* compute cell integrals:",
          np.round(ttoc - ttic, decimals=1), "sec")

    print("Save to dataset...")

    ebar = (np.multiply(nmap, emap) + 4 * eint) / (4 + nmap)
    sbar = (np.multiply(nmap, smap) + 4 * sint) / (4 + nmap)
    xbar = (np.multiply(nmap, xmap) + 4 * xint) / (4 + nmap)
    ybar = (np.multiply(nmap, ymap) + 4 * yint) / (4 + nmap)
    obar = (np.multiply(nmap, omap) + 4 * oint) / (4 + nmap)
    ibar = (np.multiply(nmap, imap) + 4 * iint) / (4 + nmap)

    obar = np.maximum(0., np.minimum(obar, -ebar))

    tfrc = np.zeros(ofrc.shape, dtype=np.float32)
    tfrc[ebar <= +0.] = -ebar[ebar <= +0.]
    tfrc[ebar <= -1.] = 1.
    ofrc = (np.multiply(nmap, ofrc) + 4 * tfrc) / (4 + nmap)

    tfrc = np.zeros(ofrc.shape, dtype=np.float32)
    tfrc[ibar >= +0.] = +ibar[ibar >= +0.]
    tfrc[ibar >= +1.] = 1.
    ifrc = (np.multiply(nmap, ifrc) + 4 * tfrc) / (4 + nmap)

    if ("bed_elevation" not in mesh.variables.keys()):
        mesh.createVariable("bed_elevation", "f4", ("nCells"))

    if ("bed_slope" not in mesh.variables.keys()):
        mesh.createVariable("bed_slope", "f4", ("nCells"))

    if ("bed_dz_dx" not in mesh.variables.keys()):
        mesh.createVariable("bed_dz_dx", "f4", ("nCells"))

    if ("bed_dz_dy" not in mesh.variables.keys()):
        mesh.createVariable("bed_dz_dy", "f4", ("nCells"))

    if ("ocn_thickness" not in mesh.variables.keys()):
        mesh.createVariable("ocn_thickness", "f4", ("nCells"))

    if ("ice_thickness" not in mesh.variables.keys()):
        mesh.createVariable("ice_thickness", "f4", ("nCells"))

    mesh["bed_elevation"].units = "m"
    mesh["bed_elevation"][:] = ebar
    mesh["bed_elevation"].long_name = names.bed_elevation

    mesh["ocn_thickness"].units = "m"
    mesh["ocn_thickness"][:] = obar
    mesh["ocn_thickness"].long_name = names.ocn_thickness

    mesh["ice_thickness"].units = "m"
    mesh["ice_thickness"][:] = ibar
    mesh["ice_thickness"].long_name = names.ice_thickness

    mesh["bed_dz_dx"][:] = xbar
    mesh["bed_dz_dx"].long_name = names.bed_dz_dx
    mesh["bed_dz_dy"][:] = ybar
    mesh["bed_dz_dy"].long_name = names.bed_dz_dy

    mesh["bed_slope"].units = "deg"
    mesh["bed_slope"][:] = \
        np.arctan(sbar) * 180. / np.pi  # degrees, for ELM
    mesh["bed_slope"].long_name = names.bed_slope_deg

    if ("ocn_cover" not in mesh.variables.keys()):
        mesh.createVariable("ocn_cover", "f4", ("nCells"))

    if ("ice_cover" not in mesh.variables.keys()):
        mesh.createVariable("ice_cover", "f4", ("nCells"))

    mesh["ocn_cover"][:] = ofrc
    mesh["ocn_cover"].long_name = names.ocn_cover
    mesh["ice_cover"][:] = ifrc
    mesh["ice_cover"].long_name = names.ice_cover

    del emap
    del eint
    del smap
    del sint
    del xmap
    del xint
    del ymap
    del yint
    del omap
    del oint
    del imap
    del iint
    del ofrc
    del ifrc

    # -- Also compute profiles (ie. histograms) of elev. outputs
    # -- per cell, dividing distributions of DEM pixel
    # -- values into NLEV-1 bands. Write band endpoints to file.

    if (elev_band <= 0):
        return

    print("Eval. histogram...")

    vals = np.asarray(
        elev["bed_elevation"][:], dtype=np.float32)
    vals = np.reshape(vals, (vals.size))

    ttic = time.time()
    eprf = cell_prfl(
        mesh, smat, NLEV, vals, evrt, ecel, ebar)
    ttoc = time.time()
    print("* compute elev. profiles:",
          np.round(ttoc - ttic, decimals=1), "sec")

    vals = np.asarray(
        elev["bed_slope"][:], dtype=np.float32)
    vals = np.reshape(vals, (vals.size))

    ttic = time.time()
    sprf = cell_prfl(
        mesh, smat, NLEV, vals, svrt, scel, sbar)
    ttoc = time.time()
    print("* compute elev. profiles:",
          np.round(ttoc - ttic, decimals=1), "sec")

    vals = np.asarray(
        elev["ocn_thickness"][:], dtype=np.float32)
    vals = np.reshape(vals, (vals.size))

    ttic = time.time()
    oprf = cell_prfl(
        mesh, smat, NLEV, vals, ovrt, ocel, obar)
    ttoc = time.time()
    print("* compute elev. profiles:",
          np.round(ttoc - ttic, decimals=1), "sec")

    vals = np.asarray(
        elev["ice_thickness"][:], dtype=np.float32)
    vals = np.reshape(vals, (vals.size))

    ttic = time.time()
    iprf = cell_prfl(
        mesh, smat, NLEV, vals, ivrt, icel, ibar)
    ttoc = time.time()
    print("* compute elev. profiles:",
          np.round(ttoc - ttic, decimals=1), "sec")

    print("Save to dataset...")

    if ("nProfiles" not in mesh.dimensions.keys()):
        mesh.createDimension("nProfiles", NLEV)

    if ("bed_elevation_profile" not in mesh.variables.keys()):
        mesh.createVariable("bed_elevation_profile",
                            "f4", ("nCells", "nProfiles"))

    if ("bed_slope_profile" not in mesh.variables.keys()):
        mesh.createVariable("bed_slope_profile",
                            "f4", ("nCells", "nProfiles"))

    if ("ocn_thickness_profile" not in mesh.variables.keys()):
        mesh.createVariable("ocn_thickness_profile",
                            "f4", ("nCells", "nProfiles"))

    if ("ice_thickness_profile" not in mesh.variables.keys()):
        mesh.createVariable("ice_thickness_profile",
                            "f4", ("nCells", "nProfiles"))

    mesh["bed_elevation_profile"].units = "m"
    mesh["bed_elevation_profile"][:, :] = eprf
    mesh["bed_elevation_profile"].long_name = names.bed_elevation_profile

    mesh["bed_slope_profile"][:, :] = sprf
    mesh["bed_slope_profile"].long_name = names.bed_slope_profile

    mesh["ocn_thickness_profile"].units = "m"
    mesh["ocn_thickness_profile"][:, :] = oprf
    mesh["ocn_thickness_profile"].long_name = names.ocn_thickness_profile

    mesh["ice_thickness_profile"].units = "m"
    mesh["ice_thickness_profile"][:, :] = iprf
    mesh["ice_thickness_profile"].long_name = names.ice_thickness_profile

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

    parser.add_argument(
        "--elev-band", dest="elev_band", type=int,
        default=0,
        required=False, help="Elev. profile band(s).")

    dem_remap(parser.parse_args())

    args = parser.parse_args()
    elev_file = args.elev_file
    mpas_file = args.mpas_file
    elev_band = args.elev_band
    dem_remap(elev_file, mpas_file, elev_band)
