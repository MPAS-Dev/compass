from compass.testcase import TestCase
from compass.ocean.tests.internal_wave.initial_state import InitialState
from compass.ocean.tests.internal_wave.forward import Forward
from compass.ocean.tests.internal_wave.viz import Viz
from compass.validate import compare_variables


class TenDayTest(TestCase):
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
        name = 'ten_day_test'
        super().__init__(test_group=test_group, name=name)
        self.add_step(InitialState(test_case=self))

        step = Forward(test_case=self, ntasks=4, openmp_threads=1)
        step.add_namelist_file(
            'compass.ocean.tests.internal_wave.ten_day_test',
            'namelist.forward')
        step.add_streams_file(
            'compass.ocean.tests.internal_wave.ten_day_test',
            'streams.forward')
        self.add_step(step)

        self.add_step(Viz(test_case=self), run_by_default=False)

    def validate(self):
        """
        Validate variables against a baseline
        """
        compare_variables(test_case=self,
                          variables=['layerThickness', 'normalVelocity'],
                          filename1='forward/output.nc')
