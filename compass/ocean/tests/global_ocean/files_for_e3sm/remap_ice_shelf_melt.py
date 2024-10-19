from compass.ocean.tests.global_ocean.files_for_e3sm.files_for_e3sm_step import (  # noqa: E501
    FilesForE3SMStep,
)
from compass.ocean.tests.global_ocean.init.remap_ice_shelf_melt import (
    remap_paolo,
)


class RemapIceShelfMelt(FilesForE3SMStep):
    """
    A step for for remapping observed melt rates to the MPAS grid and staging
    them in ``assembled_files``

    Attributes
    ----------
    init : compass.ocean.tests.global_ocean.init.Init
        The test case that produces the initial condition for this run
    """  # noqa: E501
    def __init__(self, test_case, init):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        init : compass.ocean.tests.global_ocean.init.Init
            The test case that produces the initial condition for this run
        """
        super().__init__(test_case, name='remap_ice_shelf_melt', ntasks=512,
                         min_tasks=1)
        self.init = init

    def setup(self):
        """
        setup input and output files based on config options
        """
        super().setup()

        filename = 'prescribed_ismf_paolo2023.nc'

        if self.init is None:
            self.add_input_file(
                filename='Paolo_2023_ANT_G1920V01_IceShelfMelt.nc',
                target='Paolo_2023_ANT_G1920V01_IceShelfMelt.nc',
                database='initial_condition_database',
                url='https://its-live-data.s3.amazonaws.com/height_change/Antarctica/Floating/ANT_G1920V01_IceShelfMelt.nc')    # noqa: E501
            self.add_output_file(filename=filename)
        else:
            if 'remap_ice_shelf_melt' not in self.init.steps:
                raise ValueError('Something seems to be misconfigured. No '
                                 'remap_ice_shelf_melt step found in init '
                                 'test case.')
            melt_path = \
                self.init.steps['remap_ice_shelf_melt'].path

            self.add_input_file(
                filename=filename,
                work_dir_target=f'{melt_path}/{filename}')

    def run(self):
        """
        Run this step of the test case
        """
        super().run()

        if not self.with_ice_shelf_cavities or self.init is not None:
            return

        logger = self.logger
        config = self.config
        ntasks = self.ntasks
        in_filename = 'Paolo_2023_ANT_G1920V01_IceShelfMelt.nc'
        remapped_filename = 'prescribed_ismf_paolo2023.nc'

        parallel_executable = config.get('parallel', 'parallel_executable')

        base_mesh_filename = 'base_mesh.nc'
        culled_mesh_filename = 'initial_state.nc'
        mesh_name = self.mesh_short_name
        land_ice_mask_filename = 'initial_state.nc'

        remap_paolo(in_filename, base_mesh_filename,
                    culled_mesh_filename, mesh_name,
                    land_ice_mask_filename, remapped_filename,
                    logger=logger, mpi_tasks=ntasks,
                    parallel_executable=parallel_executable)
