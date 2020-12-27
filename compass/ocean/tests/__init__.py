from compass.ocean.tests import baroclinic_channel, global_ocean, \
    ice_shelf_2d, ziso


def collect():
    """
    Get a list of testcases in this configuration

    Returns
    -------
    testcases : list
        A dictionary of configurations within this core

    """
    testcases = list()
    for configuration in [baroclinic_channel, global_ocean, ice_shelf_2d,
                          ziso]:
        testcases.extend(configuration.collect())

    return testcases
