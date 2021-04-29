from compass.testcase import TestCase
from compass.ocean.tests.baroclinic_channel.initial_state import InitialState
from compass.ocean.tests.baroclinic_channel.forward import Forward
from compass.ocean.tests import baroclinic_channel
from compass.validate import compare_variables


class ThreadsTest(TestCase):
    """
    A thread test case for the baroclinic channel test group, which makes sure
    the model produces identical results with 1 and 2 threads.

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
        name = 'threads_test'
        self.resolution = resolution
        subdir = '{}/{}'.format(resolution, name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(
            InitialState(test_case=self, resolution=resolution))

        for threads in [1, 2]:
            name = '{}thread'.format(threads)
            self.add_step(
                Forward(test_case=self, name=name, subdir=name, cores=4,
                        threads=threads, resolution=resolution))

    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        baroclinic_channel.configure(self.resolution, self.config)

    def run(self):
        """
        Run each step of the test case
        """
        # run the steps
        super().run()

        # perform validation
        variables = ['temperature', 'salinity', 'layerThickness',
                     'normalVelocity']
        steps = self.steps_to_run
        if '1thread' in steps and '2thread' in steps:
            compare_variables(test_case=self, variables=variables,
                              filename1='1thread/output.nc',
                              filename2='2thread/output.nc')
