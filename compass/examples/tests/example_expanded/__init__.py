from compass.examples.tests.example_expanded.res1km import test1 as \
    res1km_test1
from compass.examples.tests.example_expanded.res1km import test2 as \
    res1km_test2
from compass.examples.tests.example_expanded.res2km import test1 as \
    res2km_test1
from compass.examples.tests.example_expanded.res2km import test2 as \
    res2km_test2


# "collect" information about each testcase in the "example_expanded"
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
    for test in [res1km_test1, res1km_test2, res2km_test1, res2km_test2]:
        testcases.append(test.collect())

    return testcases

