from compass.ocean.tests import baroclinic_channel


def collect():
    """
    Get a list of testcases in this configuration

    Returns
    -------
    testcases : list
        A dictionary of configurations within this core

    """
    testcases = list()
    for configuration in [baroclinic_channel]:
        testcases.extend(configuration.collect())

    return testcases
