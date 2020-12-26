from compass.examples.tests.example_compact import test1, test2


# "collect" information about each testcase in the "example_compact"
# configuration, including any parameters ("resolution" in this example) that
# distinguish different test cases in this configuration
def collect():
    """
    Get a list of testcases in this configuration

    Returns
    -------
    testcases : list
        A list of tests within this configuration
    """
    # Get a list of information about the testcases in this configuration.
    # In this example, each testcase (test1 and test2) has a version at each
    # of two resolutions (1km and 2km), so this configuration has 4 testcases
    # in total.
    testcases = list()
    for resolution in ['1km', '2km']:
        for test in [test1, test2]:
            testcases.append(test.collect(resolution=resolution))

    return testcases
