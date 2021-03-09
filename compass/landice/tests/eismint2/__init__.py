from compass.testcase import add_testcase
from compass.landice.tests.eismint2 import standard_experiments, \
    decomposition_test


def collect():
    """
    Get a list of test cases in this configuration

    Returns
    -------
    testcases : list
        A list of tests within this configuration
    """
    testcases = list()
    for test in [standard_experiments, decomposition_test]:
        add_testcase(testcases, test)

    return testcases
