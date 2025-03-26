import os

import xarray

from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm.files_for_e3sm_step import (  # noqa: E501
    FilesForE3SMStep,
)


class SeaiceInitialCondition(FilesForE3SMStep):
    """
    A step for creating an E3SM sea-ice initial condition from variables from
    an MPAS-Ocean restart file
    """
    def __init__(self, test_case):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.files_for_e3sm.FilesForE3SM
            The test case this step belongs to
        """  # noqa: E501

        super().__init__(test_case, name='seaice_initial_condition')

        # for now, we won't define any outputs because they include the mesh
        # short name, which is not known at setup time.  Currently, this is
        # safe because no other steps depend on the outputs of this one.

    def run(self):
        """
        Run this step of the testcase
        """
        super().run()

        dest_filename = \
            f'mpassi.{self.mesh_short_name}.{self.creation_date}.nc'

        keep_vars = self.mesh_vars + ['fCell', 'fEdge', 'fVertex']

        if self.with_ice_shelf_cavities:
            keep_vars.append('landIceMask')

        with xarray.open_dataset('restart.nc') as ds:
            ds.load()
            ds = ds[keep_vars]
            self.write_netcdf(ds, dest_filename)

        symlink(os.path.abspath(dest_filename),
                f'{self.seaice_inputdata_dir}/{dest_filename}')
