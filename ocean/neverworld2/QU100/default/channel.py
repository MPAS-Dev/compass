
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import jigsawpy


def channel():
    """
    CHANNEL: mesh a "channel" inscribed on the sphere, such
    that positions of the triangle vertices match along the
    left- and right-hand channel edges.  

    """

    SPHERE_RADIUS = +6371.0

    opts = jigsawpy.jigsaw_jig_t()
    geom = jigsawpy.jigsaw_msh_t()
    spac = jigsawpy.jigsaw_msh_t()
    init = jigsawpy.jigsaw_msh_t()
    mesh = jigsawpy.jigsaw_msh_t()

#-- setup some (arbitrary) mesh-spacing distribution 

    spac.mshID = "ellipsoid-grid"
    spac.radii = SPHERE_RADIUS * np.ones(3)
    spac.xgrid = np.linspace(
        -1.0 * np.pi, +1.0 * np.pi, 360)
    spac.ygrid = np.linspace(
        -0.5 * np.pi, +0.5 * np.pi, 180)

    hlat = 100. * np.cos(spac.ygrid) ** 2 + 100.

    hlat = np.reshape(hlat, (spac.ygrid.size, 1))

    spac.value = np.tile(hlat, (1, spac.xgrid.size))

    jigsawpy.savevtk("spac.vtk", spac)


#-- 1. MESH THE CHANNEL LEFT/RIGHT "EDGE" 

    opts.jcfg_file = "opts.jig"
    opts.geom_file = "geom.msh"
    opts.hfun_file = "spac.msh"
    opts.mesh_file = "mesh.msh"

#-- set-up geometry as left channel edge
#-- jigsaw edges defined in sph. geometry are great circles

    geom.mshID = "ellipsoid-mesh"
    geom.radii = SPHERE_RADIUS * np.ones(3)
    divs = 1024
    add_lat_arc(geom, 
                +0.0 * np.pi / 180., 
                -70. * np.pi / 180., 
                +70. * np.pi / 180., divs)
    
    jigsawpy.savemsh(opts.geom_file, geom)
    jigsawpy.savemsh(opts.hfun_file, spac)

#-- mesh arcs in the geometry (as edges)

    opts.geom_feat = True  # capture "sharp corners"
    opts.mesh_top1 = True
    opts.mesh_dims = 1  # only 1-dim cells => edges

    jigsawpy.cmd.jigsaw(opts, init)

    jigsawpy.savevtk("init.vtk", init)


#-- 2. MESH THE FULL CHANNEL USING IC.'s

    opts.jcfg_file = "opts.jig"
    opts.geom_file = "geom.msh"
    opts.hfun_file = "spac.msh"
    opts.mesh_file = "mesh.msh"
    opts.init_file = "init.msh"

#-- set-up geometry as four channel arcs

    geom.mshID = "ellipsoid-mesh"
    geom.radii = SPHERE_RADIUS * np.ones(3)
    divs = 1024
    add_lon_arc(geom, 
                -70. * np.pi / 180., 
                +0.0 * np.pi / 180., 
                +60. * np.pi / 180., divs)
    add_lat_arc(geom, 
                +60. * np.pi / 180., 
                -70. * np.pi / 180., 
                +70. * np.pi / 180., divs)
    add_lon_arc(geom, 
                +70. * np.pi / 180., 
                +60. * np.pi / 180., 
                +0.0 * np.pi / 180., divs)
    add_lat_arc(geom, 
                +0.0 * np.pi / 180., 
                +70. * np.pi / 180., 
                -70. * np.pi / 180., divs)

    jigsawpy.savemsh(opts.geom_file, geom)

#-- set-up an "initial condition" object
#-- copy fixed points / edges from left- to right-hand edge

    apos = jigsawpy.R3toS2(
        geom.radii, init.point["coord"])

    apos[:, 0] = apos[:, 0] + 60. * np.pi / 180.

    posL = init.point
    posR = np.copy(posL)
    posR["coord"] = jigsawpy.S2toR3(geom.radii, apos)

    idxL = init.edge2
    idxR = np.copy(idxL)
    idxR["index"] = idxR["index"] + posL.size

#-- jigsaw treats points / cells with IDtag < 0 as "fixed",
#-- meaning that it won't refine and / or move them within
#-- the various meshing steps...

    idxL["IDtag"] = -1
    idxR["IDtag"] = -1

    init.point = np.concatenate((posL, posR))
    init.edge2 = np.concatenate((idxL, idxR))

    jigsawpy.savemsh(opts.init_file, init)

#-- mesh full geom., starting from INIT.

    opts.geom_feat = True
    opts.mesh_top1 = True
    opts.mesh_dims = 2  # 2-dim cells now => triangles

    jigsawpy.cmd.jigsaw(opts, mesh)


#-- 3. EXTRACT CHANNEL "PART" FROM FULL SPHERICAL MESH

    segment(mesh)

#-- in this case (based on the geometry) there are two
#-- "connected regions" in the mesh...
#-- this comes through as TRIA3["IDtag"] = region ID

   #mesh.tria3 = mesh.tria3[mesh.tria3["IDtag"] == 0]
    mesh.tria3 = mesh.tria3[mesh.tria3["IDtag"] == 1]


#-- Finally, a mesh of the channel, with matching points + 
#-- edges at left-, right-hand boundaries

    jigsawpy.savevtk("mesh.vtk", mesh)

    return


def segment(mesh):
    """
    SEGMENT: mark the cells in MESH via "connected regions",
    modifies MESH in-place such that MESH.TRIA3["IDtag"] is
    an (integer) region ID number.

    Region "boundaries" are given via the set of constraint 
    edges in the mesh: MESH.EDGE2.

    """

    tidx = np.reshape(np.arange(
        0, mesh.tria3.size), (mesh.tria3.size, 1))

    ee12 = np.sort(
        mesh.tria3["index"][:, (0, 1)], axis=1)
    ee23 = np.sort(
        mesh.tria3["index"][:, (1, 2)], axis=1)
    ee31 = np.sort(
        mesh.tria3["index"][:, (2, 0)], axis=1)

    edge = np.concatenate((
        np.hstack((ee12, tidx)),
        np.hstack((ee23, tidx)),
        np.hstack((ee31, tidx))
    ))

    edge = edge[edge[:, 1].argsort(kind="stable"), :]
    edge = edge[edge[:, 0].argsort(kind="stable"), :]


    maps = np.zeros(
        (edge.shape[0] // 2, 4), dtype=edge.dtype)

    maps[:, (0, 1)] = edge[:-1:2, (0, 1)]
    maps[:, 2] = edge[:-1:2, 2]
    maps[:, 3] = edge[+1::2, 2]

    ebnd = np.sort(
        mesh.edge2["index"][:, (0, 1)], axis=1)

    bnds = np.logical_and.reduce((
        np.isin(maps[:, 0], ebnd[:, 0]),
        np.isin(maps[:, 1], ebnd[:, 1])
    ))

    maps = maps[np.logical_not(bnds.flatten()), :]


    rows = np.concatenate((maps[:, 2], maps[:, 3]))
    cols = np.concatenate((maps[:, 3], maps[:, 2]))
    vals = np.ones((rows.size), dtype=int)

    smat = csr_matrix((vals, (rows, cols)))

    cnum, cidx = connected_components(
        smat, directed=False, return_labels=True)


    mesh.tria3["IDtag"] = cidx

    return


def add_lat_arc(mesh, alon, lat0, lat1, step):
    """
    ADD-LAT-ARC: add an arc segment to MESH at prescribed lon.

    """

    if (mesh.mshID.lower() == "ellipsoid-mesh"):

        vert = np.zeros(
            step + 0, dtype=mesh.VERT2_t)
        edge = np.zeros(
            step - 1, dtype=mesh.EDGE2_t)

        next = mesh.vert2.size

        vert["coord"][:, 0] = \
            alon * np.ones(step)
        vert["coord"][:, 1] = \
            np.linspace(lat0, lat1, step)

        edge["index"][:, 0] = \
            np.arange(next + 0, next + step - 1)
        edge["index"][:, 1] = \
            np.arange(next + 1, next + step - 0)

        mesh.vert2 = \
            np.concatenate((mesh.vert2, vert))
        mesh.edge2 = \
            np.concatenate((mesh.edge2, edge))

    else:
        raise Exception(
            "ADD-LAT-ARC: invalid MESH.MSHID!")

    return


def add_lon_arc(mesh, alat, lon0, lon1, step):
    """
    ADD-LON-ARC: add an arc segment to MESH at prescribed lat.

    """

    if (mesh.mshID.lower() == "ellipsoid-mesh"):

        vert = np.zeros(
            step + 0, dtype=mesh.VERT2_t)
        edge = np.zeros(
            step - 1, dtype=mesh.EDGE2_t)

        next = mesh.vert2.size

        vert["coord"][:, 0] = \
            np.linspace(lon0, lon1, step)
        vert["coord"][:, 1] = \
            alat * np.ones(step)
        
        edge["index"][:, 0] = \
            np.arange(next + 0, next + step - 1)
        edge["index"][:, 1] = \
            np.arange(next + 1, next + step - 0)

        mesh.vert2 = \
            np.concatenate((mesh.vert2, vert))
        mesh.edge2 = \
            np.concatenate((mesh.edge2, edge))

    else:
        raise Exception(
            "ADD-LON-ARC: invalid MESH.MSHID!")

    return


if (__name__ == "__main__"): channel()
