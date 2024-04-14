
import numpy as np
import xarray as xr
from mpas_tools.cime.constants import constants
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call

from compass.step import Step


class SshFromSurfaceDensity(Step):
    """
    Compute the sea surface height that is in approximate hydrostatic balance
    with a given land-ice pressure using the density in the top layer of the
    ocean

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.mesh.MeshStep
        The step for creating the mesh
    """
    def __init__(self, test_case, init_path, name, subdir):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.init.Init
            The test case this step belongs to

        init_path : str
            The path to the initial state from which to get the land ice
            pressure and surface density

        name : str
            The name of the step

        subdir : str
            The subdirectory for the step
        """
        super().__init__(test_case=test_case, name=name, subdir=subdir)
        self.mesh = test_case.mesh
        self.add_input_file(
            filename='init.nc',
            work_dir_target=f'{init_path}/initial_state.nc')

        mesh_path = self.mesh.get_cull_mesh_path()

        self.add_input_file(
            filename='mesh.nc',
            work_dir_target=f'{mesh_path}/culled_mesh.nc')

        self.add_input_file(
            filename='original_topograpy.nc',
            work_dir_target=f'{mesh_path}/topography_culled.nc')

        self.add_output_file(filename='topography_culled.nc')

    def run(self):
        """
        Run this step of the testcase
        """
        config = self.config
        logger = self.logger

        convert_to_cdf5 = config.getboolean('ssh_adjustment',
                                            'convert_to_cdf5')

        g = constants['SHR_CONST_G']

        with xr.open_dataset('init.nc') as ds_init:
            ds_init = ds_init.isel(Time=0)
            modify_mask = np.logical_and(
                ds_init.maxLevelCell > 0,
                ds_init.sshAdjustmentMask == 1)
            land_ice_pressure = ds_init.landIcePressure

            if 'minLevelCell' in ds_init:
                min_level_cell = ds_init.minLevelCell - 1
            else:
                min_level_cell = xr.zeros_like(ds_init.maxLevelCell)

            top_density = ds_init.density.isel(nVertLevels=min_level_cell)

            ssh = xr.where(modify_mask,
                           - land_ice_pressure / (top_density * g),
                           0.)

            with xr.open_dataset('original_topograpy.nc') as ds_topo:

                ds_topo['ssh'] = ssh

                if convert_to_cdf5:
                    write_filename = 'topography_before_cdf5.nc'
                else:
                    write_filename = 'topography_culled.nc'
                write_netcdf(ds_topo, write_filename)

                if convert_to_cdf5:
                    args = ['ncks', '-O', '-5', 'topography_before_cdf5.nc',
                            'topography_culled.nc']
                    check_call(args, logger=logger)
