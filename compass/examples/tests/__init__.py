from compass.examples.tests import example_compact, example_expanded


def collect():
    """
    Get a list of testcases in this configuration

    Returns
    -------
    testcases : list
        A dictionary of configurations within this core

    """
    testcases = list()
    # make sure you add your configuration to this list so it is included
    # in the available testcases
    for configuration in [example_compact, example_expanded]:
        testcases.extend(configuration.collect())

    return testcases
