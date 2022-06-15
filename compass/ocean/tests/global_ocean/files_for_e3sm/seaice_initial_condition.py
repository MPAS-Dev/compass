import os
import xarray

from mpas_tools.io import write_netcdf

from compass.io import symlink
from compass.step import Step


class SeaiceInitialCondition(Step):
    """
    A step for creating an E3SM sea-ice initial condition from variables from
    an MPAS-Ocean restart file

    with_ice_shelf_cavities : bool
        Whether the mesh includes ice-shelf cavities
    """
    def __init__(self, test_case, restart_filename, with_ice_shelf_cavities):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.files_for_e3sm.FilesForE3SM
            The test case this step belongs to

        restart_filename : str
            A restart file from the end of the dynamic adjustment test case to
            use as the basis for an E3SM initial condition

        with_ice_shelf_cavities : bool
            Whether the mesh includes ice-shelf cavities
        """

        super().__init__(test_case, name='seaice_initial_condition', ntasks=1,
                         min_tasks=1, openmp_threads=1)

        self.add_input_file(filename='README', target='../README')
        self.add_input_file(filename='restart.nc',
                            target=f'../{restart_filename}')

        self.with_ice_shelf_cavities = with_ice_shelf_cavities

        # for now, we won't define any outputs because they include the mesh
        # short name, which is not known at setup time.  Currently, this is
        # safe because no other steps depend on the outputs of this one.

    def run(self):
        """
        Run this step of the testcase
            """
        with_ice_shelf_cavities = self.with_ice_shelf_cavities

        with xarray.open_dataset('restart.nc') as ds:
            mesh_short_name = ds.attrs['MPAS_Mesh_Short_Name']
            mesh_prefix = ds.attrs['MPAS_Mesh_Prefix']
            prefix = f'MPAS_Mesh_{mesh_prefix}'
            creation_date = ds.attrs[f'{prefix}_Version_Creation_Date']

        assembled_dir = f'../assembled_files/inputdata/ice/mpas-seaice/' \
                        f'{mesh_short_name}'
        try:
            os.makedirs(assembled_dir)
        except OSError:
            pass

        dest_filename = f'mpassi.{mesh_short_name}.{creation_date}.nc'

        keep_vars = [
            'areaCell', 'cellsOnCell', 'edgesOnCell', 'fCell', 'indexToCellID',
            'latCell', 'lonCell', 'meshDensity', 'nEdgesOnCell',
            'verticesOnCell', 'xCell', 'yCell', 'zCell', 'angleEdge',
            'cellsOnEdge', 'dcEdge', 'dvEdge', 'edgesOnEdge', 'fEdge',
            'indexToEdgeID', 'latEdge', 'lonEdge', 'nEdgesOnCell',
            'nEdgesOnEdge', 'verticesOnEdge', 'weightsOnEdge', 'xEdge',
            'yEdge', 'zEdge', 'areaTriangle', 'cellsOnVertex', 'edgesOnVertex',
            'fVertex', 'indexToVertexID', 'kiteAreasOnVertex', 'latVertex',
            'lonVertex', 'xVertex', 'yVertex', 'zVertex']

        if with_ice_shelf_cavities:
            keep_vars.append('landIceMask')

        with xarray.open_dataset('restart.nc') as ds:
            ds.load()
            ds = ds[keep_vars]
            write_netcdf(ds, dest_filename)

        symlink(f'../../../../../seaice_initial_condition/{dest_filename}',
                f'{assembled_dir}/{dest_filename}')
