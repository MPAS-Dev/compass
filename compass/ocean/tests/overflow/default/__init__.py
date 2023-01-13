from compass.testcase import TestCase
from compass.ocean.tests.overflow.initial_state import InitialState
from compass.ocean.tests.overflow.forward import Forward
from compass.ocean.tests import overflow
from compass.validate import compare_variables


class Default(TestCase):
    """
    The default test case for the overflow test

    Attributes
    ----------
    resolution : str
        The resolution of the test case
    """

    def __init__(self, test_group, resolution):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.overflow.Overflow
            The test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='default',
                         subdir=f'{resolution}/default')
        self.resolution = resolution
        self.add_step(InitialState(test_case=self))
        self.add_step(Forward(test_case=self, ntasks=4, openmp_threads=1))

    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        overflow.configure(self.resolution, self.config)

    def validate(self):
        """
        Validate variables against a baseline
        """
        compare_variables(test_case=self,
                          variables=['layerThickness', 'temperature'],
                          filename1='forward/output.nc')
