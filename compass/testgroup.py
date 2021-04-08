class TestGroup:
    """
    The base class for test groups, which are collections of test cases with
    a common purpose (e.g. global ocean, baroclinic channel, Greenland, or
    EISMINT2)

    Attributes
    ----------
    name : str
        the name of the test group

    test_cases : dict
        A dictionary of test cases in the test group with the names of the
        test cases as keys

    mpas_core
    """

    def __init__(self, name):
        """
        Create a new test group

        Parameters
        ----------
        name : str
            the name of the test group
        """
        self.name = name
        self.mpas_core = None
        self.test_cases = dict()

    def add_test_case(self, test_case):
        """
        Add a test case to the test group

        Parameters
        ----------
        test_case : compass.TestCase
            The test case to add
        """
        self.test_cases[test_case.name] = test_case
        test_case.test_group = self
        for step in test_case.steps.values():
            step.test_group = self
