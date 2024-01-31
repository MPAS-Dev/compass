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

        step = RunModel(test_case=self, name=name, openmp_threads=1)

        # download and link the mesh
        step.mesh_file = 'landice_grid.nc'
        step.add_input_file(filename=step.mesh_file,
                            target='MISMIP_2km_20220502.nc',
                            database='')

        self.add_step(step)

    # no run() method is needed

    def configure(self):
        """
        Set up the directory structure, based on the requested resolution.
        Also ensure the requested resolution is supported (i.e. that a spun-up
        restart file exists for the resoltion).
        """
        # list of currently supported resolutions
        supported_resolutions = [2000]

        # make a formatted list (of floats) for the supported resolutions.
        # this will displayed by the `ValueError` if an unsupported
        # resolution is requested.
        supp_res_str = ", ".join([f"{x:4.0f}" for x in supported_resolutions])

        # get the config options from the TestCase, which
        config = self.config
        # get the resolution from the parsed config file(s)
        resolution = config.getfloat('mesh', 'resolution')

        # make sure the requested resolution is one that supported.
        # i.e. a resolution for which a spun-up initial condition file exists.
        if resolution not in supported_resolutions:
            raise ValueError(f'The requested resolution of {resolution:4.0f}'
                             f' (m) is not supported for the `SmokeTest`'
                             f' Supported resolutions are {supp_res_str} (m).'
                             )

        # loop over the steps of the `TestCase` and create a consitent
        # directory structure based on the value of `resolution` at the time
        # of compass setup.
        for step in self.steps.values():
            mismipplus.configure(step, config, resolution)

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
        output_path = self.steps["smoke_test"].work_dir

        compare_variables(test_case=self, variables=variables,
                          filename1=f'{output_path}/output.nc',
                          filename2=f'{output_path}/output.nc')
