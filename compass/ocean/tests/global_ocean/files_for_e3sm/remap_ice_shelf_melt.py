import os

from compass.io import symlink
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
        """  # noqa: E501
        super().__init__(test_case, name='remap_ice_shelf_melt', ntasks=512,
                         min_tasks=1)
        self.init = init
        filename = 'prescribed_ismf_paolo2023.nc'
        if init is None:
            self.add_input_file(
                filename='Paolo_2023_ANT_G1920V01_IceShelfMelt.nc',
                target='Paolo_2023_ANT_G1920V01_IceShelfMelt.nc',
                database='initial_condition_database',
                url='https://its-live-data.s3.amazonaws.com/height_change/Antarctica/Floating/ANT_G1920V01_IceShelfMelt.nc')    # noqa: E501
        elif 'remap_ice_shelf_melt' in self.init.steps:
            melt_path = \
                self.init.steps['remap_ice_shelf_melt'].path

            self.add_input_file(
                filename=filename,
                work_dir_target=f'{melt_path}/{filename}')

    def setup(self):
        """
        setup input files based on config options
        """
        super().setup()
        filename = 'prescribed_ismf_paolo2023.nc'
        if self.with_ice_shelf_cavities:
            self.add_output_file(filename=filename)

    def run(self):
        """
        Run this step of the test case
        """
        super().run()

        if not self.with_ice_shelf_cavities:
            return

        prefix = 'prescribed_ismf_paolo2023'
        suffix = f'{self.mesh_short_name}.{self.creation_date}'

        remapped_filename = f'{prefix}.nc'
        dest_filename = f'{prefix}.{suffix}.nc'

        if self.init is None:
            logger = self.logger
            config = self.config
            ntasks = self.ntasks
            in_filename = 'Paolo_2023_ANT_G1920V01_IceShelfMelt.nc'

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

        symlink(
            os.path.abspath(remapped_filename),
            f'{self.ocean_inputdata_dir}/{dest_filename}')
