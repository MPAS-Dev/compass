from compass.ocean.tests.global_ocean import init, performance_test
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
    for mesh_name in ['QU240']:
        for test in [init]:
            testcases.append(test.collect(mesh_name=mesh_name))

    for mesh_name in ['QU240']:
        for test in [performance_test]:
            for time_integrator in ['split_explicit', 'RK4']:
                testcases.append(test.collect(mesh_name=mesh_name,
                                              time_integrator=time_integrator))

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
    name = testcase['name']
    add_config(config, 'compass.ocean.tests.global_ocean.{}'.format(name),
               '{}.cfg'.format(name), exception=False)
