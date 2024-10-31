from compass.ocean.tests.utility.create_salin_restoring.extrap_salin import (
    ExtrapSalin,
)
from compass.ocean.tests.utility.create_salin_restoring.salinity_restoring import (  # noqa: E501
    Salinity,
)
from compass.testcase import TestCase


class CreateSalinRestoring(TestCase):
    """
    A test case for first creating monthly sea surface salinity from WOA23
    then extrapolating the WOA23 data into missing ocean regions such as
    ice-shelf cavities and coasts
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.utility.Utility
            The test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='create_salin_restoring')

        self.add_step(Salinity(test_case=self))
        self.add_step(ExtrapSalin(test_case=self))
