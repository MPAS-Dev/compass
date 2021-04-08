import os


class MpasCore:
    """
    The base class for housing all the tests for a given MPAS core, such as
    ocean, landice or sw (shallow water)

    Attributes
    ----------
    test_groups : dict
        A dictionary of test groups for the MPAS core with their names as keys
    """

    def __init__(self, name):
        """
        Create a new container for the test groups for a given MPAS core

        Parameters
        ----------
        name : str
            the name of the test group
        """
        self.test_groups = dict()
        self.name = name

    def add_test_group(self, test_group):
        """
        Add a test group to the MPAS core

        Parameters
        ----------
        test_group : compass.TestGroup
            the test group to add
        """
        self.test_groups[test_group.name] = test_group
        test_group.mpas_core = self
        for test_case in test_group.test_cases.values():
            test_case.mpas_core = self
            test_case.path = os.path.join(self.name, test_group.name,
                                          test_case.subdir)
            for step in test_case.steps.values():
                step.mpas_core = self
                step.path = os.path.join(self.name, test_group.name,
                                         test_case.subdir, step.subdir)
