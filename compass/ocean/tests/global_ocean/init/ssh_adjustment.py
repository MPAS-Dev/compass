from importlib.resources import contents

import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call

from compass.model import partition, run_model
from compass.ocean.tests.global_ocean.forward import ForwardStep


class SshAdjustment(ForwardStep):
    """
    A step for iteratively adjusting the pressure from the weight of the ice
    shelf to match the sea-surface height as part of ice-shelf 2D test cases
    """
    def __init__(self, test_case, init_path, name='ssh_adjustment',
                 subdir=None):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.init.Init
            The test case this step belongs to

        init_path : str
            The path to the initial state to use for forward runs

        name : str, optional
            The name of the step

        subdir : str, optional
            The subdirectory for the step
        """
        super().__init__(test_case=test_case, mesh=test_case.mesh,
                         time_integrator='split_explicit_ab2',
                         name=name, subdir=subdir)

        self.add_namelist_options({'config_AM_globalStats_enable': '.false.'})
        self.add_namelist_file('compass.ocean.namelists',
                               'namelist.ssh_adjust')

        self.add_streams_file('compass.ocean.tests.global_ocean.init',
                              'streams.ssh_adjust')

        mesh_package = test_case.mesh.package
        mesh_package_contents = list(contents(mesh_package))
        mesh_namelist = 'namelist.ssh_adjust'
        if mesh_namelist in mesh_package_contents:
            self.add_namelist_file(mesh_package, mesh_namelist)

        mesh_streams = 'streams.ssh_adjust'
        if mesh_streams in mesh_package_contents:
            self.add_streams_file(mesh_package, mesh_streams)

        mesh_path = test_case.mesh.get_cull_mesh_path()

        self.add_input_file(
            filename='init.nc',
            work_dir_target=f'{init_path}/initial_state.nc')
        self.add_input_file(
            filename='forcing_data.nc',
            work_dir_target=f'{init_path}/init_mode_forcing_data.nc')
        self.add_input_file(
            filename='graph.info',
            work_dir_target=f'{mesh_path}/culled_graph.info')

        self.add_input_file(
            filename='original_topograpy.nc',
            work_dir_target=f'{mesh_path}/topography_culled.nc')

        self.add_output_file(filename='topography_culled.nc')

    def run(self):
        """
        Run this step of the testcase
        """
        config = self.config
        update_pio = config.getboolean('global_ocean', 'forward_update_pio')
        convert_to_cdf5 = config.getboolean('ssh_adjustment',
                                            'convert_to_cdf5')

        self._adjust_ssh(update_pio=update_pio,
                         convert_to_cdf5=convert_to_cdf5)

    def _adjust_ssh(self, update_pio, convert_to_cdf5):
        """
        Adjust the sea surface height to be dynamically consistent with
        land-ice pressure.
        """
        ntasks = self.ntasks
        config = self.config
        logger = self.logger

        if update_pio:
            self.update_namelist_pio('namelist.ocean')
        partition(ntasks, config, logger)

        with xr.open_dataset('init.nc') as ds:
            ds = ds.isel(Time=0)
            init_ssh = ds.ssh
            modify_mask = np.logical_and(ds.maxLevelCell > 0,
                                         ds.modifyLandIcePressureMask == 1)
            land_ice_pressure = ds.landIcePressure

        logger.info("   * Running forward model")
        run_model(self, update_pio=False, partition_graph=False)
        logger.info("   - Complete")

        logger.info("   * Updating SSH")

        with xr.open_dataset('output_ssh.nc') as ds_ssh:
            # get the last time entry
            ds_ssh = ds_ssh.isel(Time=ds_ssh.sizes['Time'] - 1)
            final_ssh = ds_ssh.ssh

        delta_ssh = modify_mask * (final_ssh - init_ssh)

        with xr.open_dataset('original_topograpy.nc') as ds_out:

            ds_out['ssh'] = modify_mask * final_ssh

            if convert_to_cdf5:
                write_filename = 'topography_before_cdf5.nc'
            else:
                write_filename = 'topography_culled.nc'
            write_netcdf(ds_out, write_filename)
            if convert_to_cdf5:
                args = ['ncks', '-O', '-5', 'topography_before_cdf5.nc',
                        'topography_culled.nc']
                check_call(args, logger=logger)

            # Write the largest change in SSH and its lon/lat to a file
            with open('maxDeltaSSH.log', 'w') as log_file:

                icell = np.abs(delta_ssh).argmax().values

                ds_cell = ds.isel(nCells=icell)
                delta_ssh_max = delta_ssh.isel(nCells=icell).values

                coords = (f'lon/lat: '
                          f'{np.rad2deg(ds_cell.lonCell.values):f} '
                          f'{np.rad2deg(ds_cell.latCell.values):f}')
                string = (f'delta_ssh_max: '
                          f'{delta_ssh_max:g}, {coords}')

                logger.info(f'     {string}')
                log_file.write(f'{string}\n')
                string = (f'ssh: {final_ssh.isel(nCells=icell).values:g}, '
                          f'landIcePressure: '
                          f'{land_ice_pressure.isel(nCells=icell).values:g}')
                logger.info(f'     {string}')
                log_file.write(f'{string}\n')

        logger.info("   - Complete\n")
