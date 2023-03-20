import os

import xarray as xr
from mpas_tools.io import write_netcdf

from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm.files_for_e3sm_step import (  # noqa: E501
    FilesForE3SMStep,
)


class SeaiceMesh(FilesForE3SMStep):
    """
    A step for creating an MPAS-Seaice mesh from variables from an MPAS-Ocean
    initial state file
    """
    def __init__(self, test_case):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.files_for_e3sm.FilesForE3SM
            The test case this step belongs to
        """  # noqa: E501

        super().__init__(test_case, name='seaice_mesh')

        # for now, we won't define any outputs because they include the mesh
        # short name, which is not known at setup time.  Currently, this is
        # safe because no other steps depend on the outputs of this one.

    def run(self):
        """
        Run this step of the testcase
        """
        super().run()

        dest_filename = f'{self.mesh_short_name}.{self.creation_date}.nc'

        with xr.open_dataset('initial_state.nc') as ds:

            keep_vars = self.mesh_vars
            ds = ds[keep_vars]
            ds.load()
            write_netcdf(ds, dest_filename)

        symlink(os.path.abspath(dest_filename),
                f'{self.seaice_mesh_dir}/{dest_filename}')
