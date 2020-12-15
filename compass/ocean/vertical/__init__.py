import numpy
from importlib import resources
import json


def generate_grid(grid_type, vert_levels=None, max_depth=None):
    """

    Parameters
    ----------
    grid_type : {'uniform', '60layerPHC', '42layerWOCE', '100layerE3SMv1'}
        The type of vertical grid

    vert_levels : int, optional
        The number of vertical levels (one less than the number of interfaces)
        if the ``grid_type`` does not had a hard-coded number of levels

    max_depth : float, optional
        The maximum depth of the grid unless, only used for certain
        values of ``grid_type``

    Returns
    -------
    interfaces : numpy.ndarray
        A 1D array of positive depths for layer interfaces in meters

    """
    if grid_type == 'uniform':
        interfaces = _generate_uniform(vert_levels, max_depth)
    elif grid_type in ['60layerPHC', '42layerWOCE', '100layerE3SMv1']:
        interfaces = _read_json(grid_type)
    else:
        raise ValueError('Unexpected grid type: {}'.format(grid_type))

    return interfaces


def _generate_uniform(vert_levels, max_depth):
    """ Generate uniform layer interfaces between 0 and 1 """

    if vert_levels is None:
        raise ValueError('Uniform grids require a specified number of vertical '
                         'levels')

    interfaces = numpy.linspace(0., max_depth, vert_levels+1)
    return interfaces


def _read_json(grid_type):
    """ Read the grid interfaces from a json file """

    filename = '{}.json'.format(grid_type)
    with resources.open_text("compass.ocean.vertical", filename) as data_file:
        data = json.load(data_file)
        interfaces = numpy.array(data)

    return interfaces
