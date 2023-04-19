from compass.testcase import TestCase
from compass.ocean.tests.turbulence_closure.initial_state import InitialState
from compass.ocean.tests.turbulence_closure.forward import Forward
from compass.ocean.tests import turbulence_closure
from compass.validate import compare_variables


class DecompTest(TestCase):
    """
    A decomposition test case for the turbulence closure test group, which
    makes sure the model produces identical results on 1 and 4 cores.

    Attributes
    ----------
    resolution : float
        The resolution of the test case in meters
    """

    def __init__(self, test_group, resolution, forcing='cooling'):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.turbulence_closure.TurbulenceClosure
            The test group that this test case belongs to

        resolution : float
            The resolution of the test case in meters
        """
        name = 'decomp_test'
        self.resolution = resolution
        if resolution >= 1e3:
            res_name = f'{int(resolution/1e3)}km'
        else:
            res_name = f'{int(resolution)}m'
        subdir = f'{res_name}/{forcing}/{name}'
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
        turbulence_closure.configure(self.resolution, self.config)

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
