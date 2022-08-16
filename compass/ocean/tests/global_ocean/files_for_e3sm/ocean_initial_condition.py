import os
import xarray

from mpas_tools.io import write_netcdf

from compass.io import symlink
from compass.step import Step


class OceanInitialCondition(Step):
    """
    A step for creating an E3SM ocean initial condition from the results of
    a dynamic-adjustment process to dissipate fast waves
    """
    def __init__(self, test_case, restart_filename):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.files_for_e3sm.FilesForE3SM
            The test case this step belongs to

        restart_filename : str
            A restart file from the end of the dynamic adjustment test case to
            use as the basis for an E3SM initial condition
        """

        super().__init__(test_case, name='ocean_initial_condition', ntasks=1,
                         min_tasks=1, openmp_threads=1)

        self.add_input_file(filename='README', target='../README')
        self.add_input_file(filename='restart.nc',
                            target='../{}'.format(restart_filename))

        # for now, we won't define any outputs because they include the mesh
        # short name, which is not known at setup time.  Currently, this is
        # safe because no other steps depend on the outputs of this one.

    def run(self):
        """
        Run this step of the testcase
        """
        with xarray.open_dataset('restart.nc') as ds:
            mesh_short_name = ds.attrs['MPAS_Mesh_Short_Name']
            mesh_prefix = ds.attrs['MPAS_Mesh_Prefix']
            prefix = 'MPAS_Mesh_{}'.format(mesh_prefix)
            creation_date = ds.attrs['{}_Version_Creation_Date'.format(prefix)]

        try:
            os.makedirs('../assembled_files/inputdata/ocn/mpas-o/{}'.format(
                mesh_short_name))
        except OSError:
            pass

        source_filename = 'restart.nc'
        dest_filename = 'mpaso.{}.{}.nc'.format(mesh_short_name, creation_date)

        with xarray.open_dataset(source_filename) as ds:
            ds.load()
            ds = ds.drop_vars('xtime')
            write_netcdf(ds, dest_filename)

        symlink(
            '../../../../../ocean_initial_condition/{}'.format(dest_filename),
            '../assembled_files/inputdata/ocn/mpas-o/{}/{}'.format(
                mesh_short_name, dest_filename))
