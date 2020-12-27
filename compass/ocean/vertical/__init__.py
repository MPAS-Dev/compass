import numpy
from importlib import resources
import json
from netCDF4 import Dataset
import numpy as np
from scipy.optimize import root_scalar


def generate_grid(config):
    """
    Generate a vertical grid for a test case, using the config options in the
    ``vertical_grid`` section

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options with parameters used to construct the vertical
        grid

    Returns
    -------
    interfaces : numpy.ndarray
        A 1D array of positive depths for layer interfaces in meters
    """
    section = config['vertical_grid']
    grid_type = section.get('grid_type')
    if grid_type == 'uniform':
        vert_levels = section.getint('vert_levels')
        interfaces = _generate_uniform(vert_levels)
    elif grid_type == 'tanh_dz':
        vert_levels = section.getint('vert_levels')
        min_layer_thickness = section.getfloat('min_layer_thickness')
        max_layer_thickness = section.getfloat('max_layer_thickness')
        bottom_depth = section.getfloat('bottom_depth')
        interfaces = _create_tanh_dz_grid(vert_levels, bottom_depth,
                                          min_layer_thickness,
                                          max_layer_thickness)

    elif grid_type in ['60layerPHC', '42layerWOCE', '100layerE3SMv1']:
        interfaces = _read_json(grid_type)
    else:
        raise ValueError('Unexpected grid type: {}'.format(grid_type))

    if config.has_option('vertical_grid', 'bottom_depth') and \
            grid_type != 'tanh_dz':
        bottom_depth = section.getfloat('bottom_depth')
        # renormalize to the requested range
        interfaces = (bottom_depth/interfaces[-1]) * interfaces

    return interfaces


def write_grid(interfaces, out_filename):
    """
    write the vertical grid to a file

    Parameters
    ----------
    interfaces : numpy.ndarray
        A 1D array of positive depths for layer interfaces in meters

    out_filename : str
        MPAS file name for output of vertical grid
    """

    nz = len(interfaces) - 1

    # open a new netCDF file for writing.
    ncfile = Dataset(out_filename, 'w')
    # create the depth_t dimension.
    ncfile.createDimension('nVertLevels', nz)

    refBottomDepth = ncfile.createVariable(
        'refBottomDepth', np.dtype('float64').char, ('nVertLevels',))
    refMidDepth = ncfile.createVariable(
        'refMidDepth', np.dtype('float64').char, ('nVertLevels',))
    refLayerThickness = ncfile.createVariable(
        'refLayerThickness', np.dtype('float64').char, ('nVertLevels',))

    botDepth = interfaces[1:]
    midDepth = 0.5 * (interfaces[0:-1] + interfaces[1:])

    refBottomDepth[:] = botDepth
    refMidDepth[:] = midDepth
    refLayerThickness[:] = interfaces[1:] - interfaces[0:-1]
    ncfile.close()


def _generate_uniform(vert_levels):
    """ Generate uniform layer interfaces between 0 and 1 """
    interfaces = numpy.linspace(0., 1., vert_levels+1)
    return interfaces


def _read_json(grid_type):
    """ Read the grid interfaces from a json file """

    filename = '{}.json'.format(grid_type)
    with resources.open_text("compass.ocean.vertical", filename) as data_file:
        data = json.load(data_file)
        interfaces = numpy.array(data)

    return interfaces


def _create_tanh_dz_grid(num_vert_levels, bottom_depth, min_layer_thickness,
                         max_layer_thickness):
    """
    Creates the vertical grid for MPAS-Ocean and writes it to a NetCDF file

    Parameters
    ----------
    num_vert_levels : int
        Number of vertical levels for the grid

    bottom_depth : float
        bottom depth for the chosen vertical coordinate [m]

    min_layer_thickness : float
        Target thickness of the first layer [m]

    max_layer_thickness : float
        Target maximum thickness in column [m]

    Returns
    -------
    interfaces : numpy.ndarray
        A 1D array of positive depths for layer interfaces in meters
    """

    nz = num_vert_levels
    dz1 = min_layer_thickness
    dz2 = max_layer_thickness

    # the bracket here is large enough that it should hopefully encompass any
    # reasonable value of delta, the characteristic length scale over which
    # dz varies.  The args are passed on to the match_bottom function below,
    # and the root finder will determine a value of delta (sol.root) such that
    # match_bottom is within a tolerance of zero, meaning the bottom of the
    # coordinate computed by cumsum_z hits bottom_depth almost exactly
    sol = root_scalar(_match_bottom, method='brentq',
                      bracket=[dz1, 10 * bottom_depth],
                      args=(nz, dz1, dz2, bottom_depth))

    delta = sol.root
    layerThickness, z = _cumsum_z(delta, nz, dz1, dz2)
    interfaces = -z

    return interfaces


def _match_bottom(delta, nz, dz1, dz2, bottom_depth):
    """
    Compute the difference between the bottom depth computed with the given
    parameters and the target ``bottom_depth``, used in the root finding
    algorithm to determine which value of ``delta`` to use.

    Parameters
    ----------
    delta : float
        The characteristic length scale over which dz varies (this parameter
        will be optimized to hit a target depth in a target number of layers)

    nz : int
        The number of layers

    dz1 : float
        The layer thickness at the top of the ocean (z = 0)

    dz2 : float
        The layer thickness at z --> -infinity

    bottom_depth: float
        depth of the bottom of the ocean that should match the bottom layer
        interface.  Note: the bottom_depth is positive, whereas the layer
        interfaces are negative.

    Returns
    -------
    diff : float
        The computed bottom depth minus the target ``bottom_depth``.  ``diff``
        should be zero when we have found the desired ``delta``.
    """
    _, z = _cumsum_z(delta, nz, dz1, dz2)
    diff = -bottom_depth - z[-1]
    return diff


def _cumsum_z(delta, nz, dz1, dz2):
    """
    Compute layer interface depths and layer thicknesses over ``nz`` layers

    Parameters
    ----------
    delta : float
        The characteristic length scale over which dz varies (this parameter
        will be optimized to hit a target depth in a target number of layers)

    nz : int
        The number of layers

    dz1 : float
        The layer thickness at the top of the ocean (z = 0)

    dz2 : float
        The layer thickness at z --> -infinity

    Returns
    -------
    dz : numpy.ndarray
        The layer thicknesses for each layer

    z : numpy.ndarray
        The depth (positive up) of each layer interface (``nz + 1`` total
        elements)
    """
    dz = np.zeros(nz)
    z = np.zeros(nz + 1)
    for zindex in range(nz):
        dz[zindex] = _dz_z(z[zindex], dz1, dz2, delta)
        z[zindex + 1] = z[zindex] - dz[zindex]
    return dz, z


def _dz_z(z, dz1, dz2, delta):
    """
    layer thickness as a function of depth

    Parameters
    ----------
    z : float
        Depth coordinate (positive up) at which to find the layer thickness

    dz1 : float
        The layer thickness at the top of the ocean (z = 0)

    dz2 : float
        The layer thickness at z --> -infinity

    delta : float
        The characteristic length scale over which dz varies (this parameter
        will be optimized to hit a target depth in a target numer of layers)

    Returns
    -------
    dz : float
        The layer thickness
    """
    return (dz2 - dz1) * np.tanh(-z * np.pi / delta) + dz1
