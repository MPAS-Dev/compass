from compass.testcase import TestCase
from compass.ocean.tests.internal_wave.initial_state import InitialState
from compass.ocean.tests.internal_wave.forward import Forward
from compass.ocean.tests.internal_wave.viz import Viz
from compass.validate import compare_variables


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
        self.add_step(InitialState(test_case=self))
        self.add_step(Forward(test_case=self, ntasks=4, openmp_threads=1))
        self.add_step(Viz(test_case=self), run_by_default=False)

    def validate(self):
        """
        Validate variables against a baseline
        """
        compare_variables(test_case=self,
                          variables=['layerThickness', 'normalVelocity'],
                          filename1='forward/output.nc')
