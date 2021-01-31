from compass.ocean.tests.ice_shelf_2d import default, restart_test
from compass.config import add_config
from compass.testcase import add_testcase


def collect():
    """
    Get a list of test cases in this configuration

    Returns
    -------
    testcases : list
        A list of tests within this configuration
    """
    testcases = list()
    for resolution in ['5km']:
        for test in [default, restart_test]:
            add_testcase(testcases, test, resolution=resolution)

    return testcases


def configure(testcase, config):
    """
    Modify the configuration options for this test case

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case

    config : configparser.ConfigParser
        Configuration options for this test case
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
