from compass.testcase import TestCase
from compass.ocean.tests.internal_wave.initial_state import InitialState
from compass.ocean.tests.internal_wave.forward import Forward
from compass.ocean.tests.internal_wave.rpe_test.analysis import Analysis
from compass.ocean.tests import internal_wave


class RpeTest(TestCase):
    """
    The reference potential energy (RPE) test case for the internal wave
    test group performs a 20-day integration of the model forward in time at
    5 different values of the viscosity at the given resolution.

    Attributes
    ----------
    resolution : str
        The resolution of the test case
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

        nus = [0.01, 1, 15, 150]

        self.add_step(InitialState(test_case=self))

        for index, nu in enumerate(nus):
            name = 'rpe_test_{}_nu_{:g}'.format(index + 1, nu)
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
