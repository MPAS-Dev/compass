from compass.testcase import TestCase
from compass.ocean.tests.baroclinic_channel.initial_state import InitialState
from compass.ocean.tests.baroclinic_channel.forward import Forward
from compass.ocean.tests import baroclinic_channel
from compass.validate import compare_variables


class DecompTest(TestCase):
    """
    A decomposition test case for the baroclinic channel test group, which
    makes sure the model produces identical results on 1 and 4 cores.

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
        name = 'decomp_test'
        self.resolution = resolution
        subdir = '{}/{}'.format(resolution, name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(
            InitialState(test_case=self, resolution=resolution))

        for procs in [4, 8]:
            name = '{}proc'.format(procs)
            self.add_step(
                Forward(test_case=self, name=name, subdir=name, ntasks=procs,
                        openmp_threads=1, resolution=resolution))

    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        baroclinic_channel.configure(self.resolution, self.config)

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['temperature', 'salinity', 'layerThickness',
                     'normalVelocity']
        compare_variables(test_case=self, variables=variables,
                          filename1='4proc/output.nc',
                          filename2='8proc/output.nc')
