class MpasCore:
    """
    The base class for housing all the tests for a given MPAS core, such as
    ocean, landice or sw (shallow water)

    Attributes
    ----------
    name : str
        the name of the MPAS core

    test_groups : dict
        A dictionary of test groups for the MPAS core with their names as keys
    """

    def __init__(self, name):
        """
        Create a new container for the test groups for a given MPAS core

        Parameters
        ----------
        name : str
            the name of the MPAS core
        """
        self.name = name

        # test groups are added with add_test_groups()
        self.test_groups = dict()

    def add_test_group(self, test_group):
        """
        Add a test group to the MPAS core

        Parameters
        ----------
        test_group : compass.TestGroup
            the test group to add
        """
        self.test_groups[test_group.name] = test_group
