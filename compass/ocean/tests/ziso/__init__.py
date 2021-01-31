from compass.testcase import add_testcase
from compass.ocean.tests.ziso import default, with_frazil
from compass.config import add_config


def collect():
    """
    Get a list of test cases in this configuration

    Returns
    -------
    testcases : list
        A list of tests within this configuration
    """
    testcases = list()
    for resolution in ['20km']:
        for test in [default, with_frazil]:
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
    res_params = {'20km': {'nx': 50,
                           'ny': 112,
                           'dc': 20e3}}

    if resolution not in res_params:
        raise ValueError('Unsupported resolution {}. Supported values are: '
                         '{}'.format(resolution, list(res_params)))
    res_params = res_params[resolution]
    for param in res_params:
        config.set('ziso', param, '{}'.format(res_params[param]))

    add_config(config, 'compass.ocean.tests.ziso.{}'.format(name),
               '{}.cfg'.format(name), exception=False)
