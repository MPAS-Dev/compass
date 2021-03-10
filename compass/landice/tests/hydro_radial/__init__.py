from compass.testcase import add_testcase
from compass.landice.tests.hydro_radial import decomposition_test, \
    restart_test, spinup_test, steady_state_drift_test


def collect():
    """
    Get a list of test cases in this configuration

    Returns
    -------
    testcases : list
        A list of tests within this configuration
    """
    testcases = list()
    for test in [decomposition_test, restart_test, spinup_test,
                 steady_state_drift_test]:
        add_testcase(testcases, test)

    return testcases
