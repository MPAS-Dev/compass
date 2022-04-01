import numpy as np
import jigsawpy
from jigsawpy.savejig import savejig

from mpas_tools.logging import check_call, LoggingContext
from mpas_tools.cime.constants import constants


def from_cell_widths(cell_width, lon, lat, logger=None):
    """
    A function for building a jigsaw mesh

    Parameters
    ----------
    cell_width : ndarray
        The size of each cell in the resulting mesh as a function of space

    lon, lat : ndarray
        The x and y coordinates of each point in the cellWidth array (lon and
        lat for spherical mesh)

    logger : logging.Logger, optional
        A logger for the output if not stdout
    """

    earth_radius = constants['SHR_CONST_REARTH']

    # setup files for JIGSAW
    opts = jigsawpy.jigsaw_jig_t()
    opts.geom_file = 'mesh.msh'
    opts.jcfg_file = 'mesh.jig'
    opts.mesh_file = 'mesh-MESH.msh'
    opts.hfun_file = 'mesh-HFUN.msh'

    # save HFUN data to file
    hmat = jigsawpy.jigsaw_msh_t()
    hmat.mshID = 'ELLIPSOID-GRID'
    hmat.xgrid = np.radians(lon)
    hmat.ygrid = np.radians(lat)
    hmat.value = cell_width
    jigsawpy.savemsh(opts.hfun_file, hmat)

    # define JIGSAW geometry
    geom = jigsawpy.jigsaw_msh_t()
    geom.mshID = 'ELLIPSOID-MESH'
    geom.radii = earth_radius*1e-3*np.ones(3, float)
    jigsawpy.savemsh(opts.geom_file, geom)

    # build mesh via JIGSAW!
    opts.hfun_scal = 'absolute'
    opts.hfun_hmax = float('inf')
    opts.hfun_hmin = 0.0
    # 2-dim. simplexes
    opts.mesh_dims = 2
    opts.optm_qlim = 0.9375
    opts.verbosity = 1

    savejig(opts.jcfg_file, opts)
    check_call(['jigsaw', opts.jcfg_file], logger=logger)


def icosahetral(subdivisions, logger=None):
    """
    A function for building a jigsaw mesh

    Parameters
    ----------
    subdivisions : int
        The number of subdivisions of the icosahedron to make

    logger : logging.Logger, optional
        A logger for the output if not stdout
    """
    earth_radius = constants['SHR_CONST_REARTH']

    opts = jigsawpy.jigsaw_jig_t()

    opts.geom_file = 'mesh.msh'
    opts.mesh_file = 'mesh-MESH.msh'

    opts.hfun_hmax = 1.
    # 2-dim. simplexes
    opts.mesh_dims = 2

    opts.optm_iter = 512
    opts.optm_qtol = 1e-6

    geom = jigsawpy.jigsaw_msh_t()
    geom.mshID = 'ellipsoid-mesh'
    geom.radii = earth_radius*1e-3*np.ones(3, float)
    jigsawpy.savemsh(opts.geom_file, geom)

    with LoggingContext(__name__, logger=logger):

        icos = jigsawpy.jigsaw_msh_t()
        jigsawpy.cmd.icosahedron(opts, subdivisions, icos)
        jigsawpy.savemsh(opts.mesh_file, icos)
