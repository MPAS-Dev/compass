from compass.testcase import TestCase
from compass.ocean.tests.overflow.initial_state import InitialState
from compass.ocean.tests.overflow.forward import Forward
from compass.ocean.tests.overflow.rpe_test.analysis import Analysis
from compass.ocean.tests import overflow


class RpeTest(TestCase):
    """
    The reference potential energy (RPE) test case for the overflow
    test group performs a 40h integration of the model forward in time at
    5 different values of the viscosity at the given resolution.

    Attributes
    ----------
    resolution : str
        The resolution of the test case
    """

    def __init__(self, test_group, resolution='1km'):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.overflow.Overflow
            The test group that this test case belongs to

        """
        name = 'rpe_test'
        subdir = f'{resolution}/{name}'
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.resolution = resolution
        self.nus = None


    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        config = self.config
        resolution = self.resolution

        overflow.configure(resolution, config)

        nus = config.getlist('overflow', 'viscosities', dtype=float)
        self.nus = nus

        self.add_step(
            InitialState(test_case=self))

        for index, nu in enumerate(nus):
            name = f'rpe_test_{index + 1}_nu_{int(nu)}'
            step = Forward(
                test_case=self, name=name, subdir=name,
                ntasks=64, min_tasks=16, openmp_threads=1,
                nu=float(nu))

            step.add_namelist_file(
                'compass.ocean.tests.overflow.rpe_test',
                'namelist.forward')
            step.add_streams_file(
                'compass.ocean.tests.overflow.rpe_test',
                'streams.forward')
            self.add_step(step)

        self.add_step(
            Analysis(test_case=self, resolution=resolution, nus=nus))
    # no run() is needed because we're doing the default: running all steps
