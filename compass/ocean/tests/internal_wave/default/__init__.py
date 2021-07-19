from compass.testcase import TestCase
from compass.ocean.tests.internal_wave.default.init import Init


class Default(TestCase):
    """
    The default test case for the internal wave test
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.internal_wave.InternalWave
            The test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='default')
        self.add_step(Init(test_case=self))
