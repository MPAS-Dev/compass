from compass.testcase import TestCase


class Default(TestCase):
    """
    The default test case for the overflow test
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.overflow.Overflow
            The test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='default')
