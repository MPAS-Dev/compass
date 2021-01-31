from compass.testcase import add_testcase
from compass.examples.tests.example_compact import test1, test2


# "collect" information about each test case in the "example_compact"
# configuration, including any parameters ("resolution" in this example) that
# distinguish different test cases in this configuration
def collect():
    """
    Get a list of test cases in this configuration

    Returns
    -------
    testcases : list
        A list of tests within this configuration
    """
    # Get a list of information about the test cases in this configuration.
    # In this example, each test case (test1 and test2) has a version at each
    # of two resolutions (1km and 2km), so this configuration has 4 test cases
    # in total.
    testcases = list()
    for resolution in ['1km', '2km']:
        for test in [test1, test2]:
            # we can pass keyword argument to the test case so they get added
            # to the "testcase" dictionary and can be used throughout the test
            # case and passed ot the step
            add_testcase(testcases, test, resolution=resolution)

    return testcases
