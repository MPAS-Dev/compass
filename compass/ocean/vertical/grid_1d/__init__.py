import json
from importlib import resources

import numpy
import numpy as np
from netCDF4 import Dataset

from compass.ocean.vertical.grid_1d.index_tanh_dz import (
    create_index_tanh_dz_grid,
)
from compass.ocean.vertical.grid_1d.tanh_dz import create_tanh_dz_grid


def generate_1d_grid(config):
    """
    Generate a vertical grid for a test case, using the config options in the
    ``vertical_grid`` section

    Parameters
    ----------
    config : compass.config.CompassConfigParser
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
        interfaces = create_tanh_dz_grid(vert_levels,
                                         bottom_depth,
                                         min_layer_thickness,
                                         max_layer_thickness)

    elif grid_type == 'index_tanh_dz':
        vert_levels = section.getint('vert_levels')
        min_layer_thickness = section.getfloat('min_layer_thickness')
        max_layer_thickness = section.getfloat('max_layer_thickness')
        bottom_depth = section.getfloat('bottom_depth')
        transition_levels = section.getfloat('transition_levels')
        interfaces = create_index_tanh_dz_grid(
            vert_levels,
            bottom_depth,
            min_layer_thickness,
            max_layer_thickness,
            transition_levels)

    elif grid_type in ['60layerPHC', '80layerE3SMv1', '100layerE3SMv1']:
        interfaces = _read_json(grid_type)
    else:
        raise ValueError('Unexpected grid type: {}'.format(grid_type))

    if config.has_option('vertical_grid', 'bottom_depth') and \
            grid_type != 'tanh_dz':
        bottom_depth = section.getfloat('bottom_depth')
        # renormalize to the requested range
        interfaces = (bottom_depth / interfaces[-1]) * interfaces

    return interfaces


def write_1d_grid(interfaces, out_filename):
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


def add_1d_grid(config, ds):
    """
    Add a 1D vertical grid based on the config options in the ``vertical_grid``
    section to a mesh data set

    The following variables are added to the mesh:
    * ``refTopDepth`` - the positive-down depth of the top of each ref. level
    * ``refZMid`` - the positive-down depth of the middle of each ref. level
    * ``refBottomDepth`` - the positive-down depth of the bottom of each ref.
      level
    * ``refInterfaces`` - the positive-down depth of the interfaces between
      ref. levels (with ``nVertLevels`` + 1 elements).
    There is considerable redundancy between these variables but each is
    sometimes convenient.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options with parameters used to construct the vertical
        grid

    ds : xarray.Dataset
        A data set to add the grid variables to
    """

    interfaces = generate_1d_grid(config=config)

    ds['refTopDepth'] = ('nVertLevels', interfaces[0:-1])
    ds['refZMid'] = ('nVertLevels', -0.5 * (interfaces[1:] + interfaces[0:-1]))
    ds['refBottomDepth'] = ('nVertLevels', interfaces[1:])
    ds['refInterfaces'] = ('nVertLevelsP1', interfaces)


def _generate_uniform(vert_levels):
    """ Generate uniform layer interfaces between 0 and 1 """
    interfaces = numpy.linspace(0., 1., vert_levels + 1)
    return interfaces


def _read_json(grid_type):
    """ Read the grid interfaces from a json file """

    filename = '{}.json'.format(grid_type)
    with resources.open_text("compass.ocean.vertical", filename) as data_file:
        data = json.load(data_file)
        interfaces = numpy.array(data)

    return interfaces
