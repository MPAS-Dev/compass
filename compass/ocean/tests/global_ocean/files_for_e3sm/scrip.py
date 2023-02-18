import os

from mpas_tools.scrip.from_mpas import scrip_from_mpas

from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm.files_for_e3sm_step import (  # noqa: E501
    FilesForE3SMStep,
)


class Scrip(FilesForE3SMStep):
    """
    A step for creating SCRIP files from the MPAS-Ocean mesh
    """
    def __init__(self, test_case):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.files_for_e3sm.FilesForE3SM
            The test case this step belongs to
        """  # noqa: E501

        super().__init__(test_case, name='scrip')

    def setup(self):
        """
        setup input files based on config options
        """
        super().setup()
        self.add_output_file(filename='ocean.scrip.nc')
        with_ice_shelf_cavities = self.with_ice_shelf_cavities
        if with_ice_shelf_cavities is not None and with_ice_shelf_cavities:
            self.add_output_file(filename='ocean.mask.scrip.nc')

    def run(self):
        """
        Run this step of the testcase
        """
        super().run()
        with_ice_shelf_cavities = self.with_ice_shelf_cavities
        mesh_short_name = self.mesh_short_name
        creation_date = self.creation_date

        if with_ice_shelf_cavities:
            nomask_str = '.nomask'
        else:
            nomask_str = ''

        local_filename = 'ocean.scrip.nc'
        scrip_filename = \
            f'ocean.{mesh_short_name}{nomask_str}.scrip.{creation_date}.nc'

        scrip_from_mpas('restart.nc', local_filename)

        symlink(os.path.abspath(local_filename),
                f'{self.ocean_inputdata_dir}/{scrip_filename}')

        if with_ice_shelf_cavities:
            local_filename = 'ocean.mask.scrip.nc'
            scrip_mask_filename = \
                f'ocean.{mesh_short_name}.mask.scrip.{creation_date}.nc'
            scrip_from_mpas('restart.nc', local_filename,
                            useLandIceMask=True)

            symlink(os.path.abspath(local_filename),
                    f'{self.ocean_inputdata_dir}//{scrip_mask_filename}')
