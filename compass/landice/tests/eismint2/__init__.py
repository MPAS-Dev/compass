from compass.testcase import add_testcase
from compass.landice.tests.eismint2 import standard_experiments, \
    decomposition_test, restart_test


def collect():
    """
    Get a list of test cases in this configuration

    Returns
    -------
    testcases : list
        A list of tests within this configuration
    """
    testcases = list()
    add_testcase(testcases, standard_experiments)
    for thermal_solver in ['temperature', 'enthalpy']:
        for test in [decomposition_test, restart_test]:
            add_testcase(testcases, test, thermal_solver=thermal_solver)

    return testcases
