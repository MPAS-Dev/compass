from compass.landice.tests import dome


def collect():
    """
    Get a list of testcases in this configuration

    Returns
    -------
    testcases : list
        A dictionary of configurations within this core

    """
    testcases = list()
    for configuration in [dome]:
        testcases.extend(configuration.collect())

    return testcases
