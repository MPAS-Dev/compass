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

    def __init__(self, test_group, resolution, basal_friction='weertman'):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.mismipplus.MISMIPplus
            The test group that this test case belongs to

        resolution : float
            The resolution of the test case. Valid options are defined in the
            test group constructor.

        basal_friction : {'weertman', 'regularized_coulomb',
                          'debris_friction'}, optional
            The basal-friction variant to configure.
        """
        base_name = 'smoke_test'
        valid_basal_friction = {
            'weertman',
            'regularized_coulomb',
            'debris_friction'
        }
        if basal_friction not in valid_basal_friction:
            raise ValueError(
                f'Unsupported basal_friction "{basal_friction}". '
                f'Valid options are: {sorted(valid_basal_friction)}')

        if resolution != 2000:
            raise ValueError(
                f'Unsupported resolution "{resolution}". '
                'The only valid resolution is 2000m.')

        resolution_name = f'{resolution:0.0f}m'
        name = f'{base_name}_{resolution_name}_{basal_friction}'
        subdir = f"{base_name}/{resolution_name}/{basal_friction}"

        super().__init__(test_group=test_group, name=name, subdir=subdir)

        step_name = 'run_model'
        step = RunModel(test_case=self, name=step_name,
                resolution=resolution,
            basal_friction=basal_friction,
            update_mesh_for_basal_friction=True)

        # download and link the mesh, eventually this will need to be
        # resolution aware. ``configure`` method is probably a better place
        # for parsing and adding the correct IC file based on resolution.
        step.mesh_file = 'landice_grid.nc'
        step.add_input_file(filename=step.mesh_file,
                            target='MISMIP_2km_20220502.nc',
                            database='',
                            copy=True)

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
