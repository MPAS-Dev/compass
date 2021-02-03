from compass.testcase import add_testcase
from compass.landice.tests.dome import smoke_test, decomposition_test, \
    restart_test


def collect():
    """
    Get a list of test cases in this configuration

    Returns
    -------
    testcases : list
        A list of tests within this configuration
    """
    testcases = list()
    for mesh_type in ['2000m', 'variable_resolution']:
        for test in [smoke_test, decomposition_test, restart_test]:
            add_testcase(testcases, test, mesh_type=mesh_type)

    return testcases
