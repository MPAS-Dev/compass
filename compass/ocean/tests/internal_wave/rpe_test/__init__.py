from compass.testcase import TestCase
from compass.ocean.tests.internal_wave.initial_state import InitialState
from compass.ocean.tests.internal_wave.forward import Forward
from compass.ocean.tests.internal_wave.rpe_test.analysis import Analysis


class RpeTest(TestCase):
    """
    The reference potential energy (RPE) test case for the internal wave
    test group performs a 20-day integration of the model forward in time at
    5 different values of the viscosity at the given resolution.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.internal_wave.InternalWave
            The test group that this test case belongs to
        """
        name = 'rpe_test'
        super().__init__(test_group=test_group, name=name)


    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        config = self.config
        self.add_step(InitialState(test_case=self))

        nus = config.getlist('internal_wave', 'viscosities', dtype=float)
        for index, nu in enumerate(nus):
            name = f'rpe_test_{index + 1}_nu_{nu:g}'
            step = Forward(
                test_case=self, name=name, subdir=name, ntasks=4,
                openmp_threads=1, nu=float(nu))

            step.add_namelist_file(
                'compass.ocean.tests.internal_wave.rpe_test',
                'namelist.forward')
            step.add_streams_file(
                'compass.ocean.tests.internal_wave.rpe_test',
                'streams.forward')
            self.add_step(step)

        self.add_step(
            Analysis(test_case=self, nus=nus))
