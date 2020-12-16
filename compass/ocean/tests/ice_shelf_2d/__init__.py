from compass.ocean.tests.ice_shelf_2d import default, restart_test
from compass.config import add_config


def collect():
    """
    Get a list of testcases in this configuration

    Returns
    -------
    testcases : list
        A list of tests within this configuration
    """
    testcases = list()
    for resolution in ['5km']:
        for test in [default, restart_test]:
            testcases.append(test.collect(resolution=resolution))

    return testcases


def configure(testcase, config):
    """
    Modify the configuration options for this testcase.

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this testcase from the ``collect()``
        function

    config : configparser.ConfigParser
        Configuration options for this testcase, a combination of the defaults
        for the machine, core and configuration
    """
    resolution = testcase['resolution']
    name = testcase['name']
    res_params = {'5km': {'nx': 10, 'ny': 44, 'dc': 5e3}}

    if resolution not in res_params:
        raise ValueError('Unsupported resolution {}. Supported values are: '
                         '{}'.format(resolution, list(res_params)))
    res_params = res_params[resolution]
    for param in res_params:
        config.set('ice_shelf_2d', param, '{}'.format(res_params[param]))

    add_config(config, 'compass.ocean.tests.ice_shelf_2d.{}'.format(name),
               '{}.cfg'.format(name), exception=False)
