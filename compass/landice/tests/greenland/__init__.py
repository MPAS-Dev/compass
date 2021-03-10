from compass.testcase import add_testcase
from compass.landice.tests.greenland import smoke_test, decomposition_test, \
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
    for resolution in ['20km']:
        for test in [smoke_test, decomposition_test, restart_test]:
            add_testcase(testcases, test, resolution=resolution)

    return testcases
