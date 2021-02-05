from compass.landice.tests import dome, enthalpy_benchmark


def collect():
    """
    Get a list of testcases in this configuration

    Returns
    -------
    testcases : list
        A dictionary of configurations within this core

    """
    testcases = list()
    for configuration in [dome, enthalpy_benchmark]:
        testcases.extend(configuration.collect())

    return testcases
