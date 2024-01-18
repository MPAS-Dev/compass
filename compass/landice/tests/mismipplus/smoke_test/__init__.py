from compass.landice.tests.mismipplus.run_model import RunModel
from compass.testcase import TestCase
from compass.validate import compare_variables


class SmokeTest(TestCase):
    """
    A test case for running a smoke test of the MISMIP+ configuration

    This test case performs a forward simulation for a short duration
    using a pre-made mesh file with a 2 km MISMIP+
    spin-up that has previously been run to steady state
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.mismipplus.MISMIPplus
            The test group that this test case belongs to

        """
        name = 'smoke_test'
        super().__init__(test_group=test_group, name=name)

        resolution = '2000m'
        step = RunModel(test_case=self,
                        name=name,
                        subdir='simulation',
                        resolution=resolution,
                        openmp_threads=1)

        # download and link the mesh
        step.mesh_file = 'landice_grid.nc'
        step.add_input_file(filename=step.mesh_file,
                            target='MISMIP_2km_20220502.nc',
                            database='')

        self.add_step(step)

    # no configure() method is needed

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        # Comparing against itself to for a smoke test
        # (This allows the potential to compare against a baseline)
        variables = ['thickness', 'surfaceSpeed']
        compare_variables(test_case=self, variables=variables,
                          filename1='simulation/output.nc',
                          filename2='simulation/output.nc')
