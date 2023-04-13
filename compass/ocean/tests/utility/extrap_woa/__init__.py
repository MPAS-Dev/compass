from compass.ocean.tests.utility.extrap_woa.combine import Combine
from compass.ocean.tests.utility.extrap_woa.extrap_step import ExtrapStep
from compass.ocean.tests.utility.extrap_woa.remap_topography import (
    RemapTopography,
)
from compass.testcase import TestCase


class ExtrapWoa(TestCase):
    """
    A test case for first remapping a topography dataset to the WOA 2023 grid,
    then extrapolating the WOA23 data into missing ocean regions such as
    ice-shelf cavities, then extrapolating into grounded ice and land
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.utility.Utility
            The test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='extrap_woa')

        self.add_step(Combine(test_case=self))
        self.add_step(RemapTopography(test_case=self))
        self.add_step(ExtrapStep(test_case=self))
