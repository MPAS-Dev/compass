from compass.testcase import add_testcase
from compass.landice.tests.enthalpy_benchmark import A, B


def collect():
    """
    Get a list of test cases in this configuration

    Returns
    -------
    testcases : list
        A list of tests within this configuration
    """
    testcases = list()
    for test in [A, B]:
        add_testcase(testcases, test)

    return testcases
