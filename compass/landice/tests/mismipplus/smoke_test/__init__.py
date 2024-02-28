from compass.landice.tests import mismipplus
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

    def __init__(self, test_group, resolution):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.mismipplus.MISMIPplus
            The test group that this test case belongs to

        resolution : float
            The resolution of the test case. Valid options are defined in the
            test group constructor.
        """
        name = 'smoke_test'
        subdir = f"{name}/{resolution:4.0f}m"

        super().__init__(test_group=test_group, name=name, subdir=subdir)

        step_name = 'run_model'
        step = RunModel(test_case=self, name=step_name, resolution=resolution)

        # download and link the mesh, eventually this will need to be
        # resolution aware. ``configure`` method is probably a better place
        # for parsing and adding the correct IC file based on resolution.
        step.mesh_file = 'landice_grid.nc'
        step.add_input_file(filename=step.mesh_file,
                            target='MISMIP_2km_20220502.nc',
                            database='')

        self.add_step(step)

    # no configure() method is needed (for now)

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        # Comparing against itself to for a smoke test
        # (This allows the potential to compare against a baseline)
        variables = ['thickness', 'surfaceSpeed']

        # access the work_dir for the step, even though validation
        # operates at the case level
        output_path = self.steps["run_model"].work_dir

        compare_variables(test_case=self, variables=variables,
                          filename1=f'{output_path}/output.nc',
                          filename2=f'{output_path}/output.nc')
