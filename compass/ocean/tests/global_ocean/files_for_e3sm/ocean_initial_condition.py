import os

import xarray
from mpas_tools.io import write_netcdf

from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm.files_for_e3sm_step import (  # noqa: E501
    FilesForE3SMStep,
)
from compass.ocean.tests.global_ocean.metadata import (
    add_mesh_and_init_metadata,
)


class OceanInitialCondition(FilesForE3SMStep):
    """
    A step for creating an E3SM ocean initial condition from the results of
    a dynamic-adjustment process to dissipate fast waves

    Attributes
    ----------
    add_metadata : bool
        Whether to add metadata to the ocean initial condition
    """
    def __init__(self, test_case, add_metadata=False):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.files_for_e3sm.FilesForE3SM
            The test case this step belongs to

        add_metadata : bool, optional
            Whether to add metadata to the ocean initial condition
        """  # noqa: E501

        super().__init__(test_case, name='ocean_initial_condition')
        self.add_metadata = add_metadata

        # for now, we won't define any outputs because they include the mesh
        # short name, which is not known at setup time.  Currently, this is
        # safe because no other steps depend on the outputs of this one.

    def run(self):
        """
        Run this step of the testcase
        """
        super().run()
        source_filename = 'restart.nc'
        dest_filename = f'mpaso.{self.mesh_short_name}.{self.creation_date}.nc'

        with xarray.open_dataset(source_filename) as ds:
            ds.load()
            if 'xtime' in ds.data_vars:
                ds = ds.drop_vars('xtime')
            write_netcdf(ds, dest_filename)

        if self.add_metadata:
            add_mesh_and_init_metadata([dest_filename], self.config,
                                       source_filename)

        symlink(
            os.path.abspath(dest_filename),
            f'{self.ocean_inputdata_dir}/{dest_filename}')
