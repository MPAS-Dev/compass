from compass.ocean.tests.utility.cull_restarts.cull import Cull
from compass.testcase import TestCase


class CullRestarts(TestCase):
    """
    A test case for culling MPAS-Ocean and -Seaice restart files to exclude
    ice-shelf cavities
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.utility.Utility
            The test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='cull_restarts')

        self.add_step(Cull(test_case=self))
