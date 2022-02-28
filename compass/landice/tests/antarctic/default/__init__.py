from compass.testcase import TestCase
from compass.landice.tests.antarctic.mesh import Mesh


class Default(TestCase):
    """
    The default test case for the humboldt test group simply creates the
    mesh and initial condition.

    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.antarctic.Antarctic
            The test group that this test case belongs to

        """
        name = 'default'
        subdir = name
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(
            Mesh(test_case=self))
