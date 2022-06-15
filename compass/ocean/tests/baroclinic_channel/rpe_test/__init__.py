from compass.testcase import TestCase
from compass.ocean.tests.baroclinic_channel.initial_state import InitialState
from compass.ocean.tests.baroclinic_channel.forward import Forward
from compass.ocean.tests.baroclinic_channel.rpe_test.analysis import Analysis
from compass.ocean.tests import baroclinic_channel


class RpeTest(TestCase):
    """
    The reference potential energy (RPE) test case for the baroclinic channel
    test group performs a 20-day integration of the model forward in time at
    5 different values of the viscosity at the given resolution.

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
        test_group : compass.ocean.tests.baroclinic_channel.BaroclinicChannel
            The test group that this test case belongs to

        resolution : str
            The resolution of the test case
        """
        name = 'rpe_test'
        subdir = f'{resolution}/{name}'
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        nus = [1, 5, 10, 20, 200]

        res_params = {'1km': {'ntasks': 144, 'min_tasks': 36},
                      '4km': {'ntasks': 36, 'min_tasks': 8},
                      '10km': {'ntasks': 8, 'min_tasks': 4}}

        if resolution not in res_params:
            raise ValueError(
                f'Unsupported resolution {resolution}. Supported values are: '
                f'{list(res_params)}')

        params = res_params[resolution]

        self.resolution = resolution

        self.add_step(
            InitialState(test_case=self, resolution=resolution))

        for index, nu in enumerate(nus):
            name = 'rpe_test_{}_nu_{}'.format(index + 1, nu)
            step = Forward(
                test_case=self, name=name, subdir=name,
                ntasks=params['ntasks'], min_tasks=params['min_tasks'],
                resolution=resolution, nu=float(nu))

            step.add_namelist_file(
                'compass.ocean.tests.baroclinic_channel.rpe_test',
                'namelist.forward')
            step.add_streams_file(
                'compass.ocean.tests.baroclinic_channel.rpe_test',
                'streams.forward')
            self.add_step(step)

        self.add_step(
            Analysis(test_case=self, resolution=resolution, nus=nus))

    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        baroclinic_channel.configure(self.resolution, self.config)

    # no run() is needed because we're doing the default: running all steps
