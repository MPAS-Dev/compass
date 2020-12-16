from compass.ocean.tests import baroclinic_channel, ice_shelf_2d


def collect():
    """
    Get a list of testcases in this configuration

    Returns
    -------
    testcases : list
        A dictionary of configurations within this core

    """
    testcases = list()
    for configuration in [baroclinic_channel, ice_shelf_2d]:
        testcases.extend(configuration.collect())

    return testcases
