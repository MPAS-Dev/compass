import numpy
from importlib import resources
import json


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
    elif grid_type in ['60layerPHC', '42layerWOCE', '100layerE3SMv1']:
        interfaces = _read_json(grid_type)
    else:
        raise ValueError('Unexpected grid type: {}'.format(grid_type))

    if config.has_option('vertical_grid', 'bottom_depth'):
        bottom_depth = section.getfloat('bottom_depth')
        # renormalize to the requested range
        interfaces = (bottom_depth/interfaces[-1]) * interfaces

    return interfaces


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
