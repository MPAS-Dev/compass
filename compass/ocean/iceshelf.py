import os
import shutil
import subprocess

import numpy
import xarray
from mpas_tools.cime.constants import constants
from mpas_tools.io import write_netcdf

from compass.io import symlink
from compass.model import partition, run_model


def compute_land_ice_pressure_and_draft(ssh, modify_mask, ref_density):
    """
    Compute the pressure from and overlying ice shelf and the ice-shelf draft

    Parameters
    ----------
    ssh : xarray.DataArray
        The sea surface height (the ice draft)

    modify_mask : xarray.DataArray
        A mask that is 1 where ``landIcePressure`` can be deviate from 0

    ref_density : float
        A reference density for seawater displaced by the ice shelf

    Returns
    -------
    landIcePressure : xarray.DataArray
        The pressure from the overlying land ice on the ocean

    landIceDraft : xarray.DataArray
        The ice draft, equal to the initial ``ssh``
    """
    gravity = constants['SHR_CONST_G']
    landIcePressure = \
        modify_mask * numpy.maximum(-ref_density * gravity * ssh, 0.)
    landIceDraft = ssh
    return landIcePressure, landIceDraft


def adjust_ssh(variable, iteration_count, step, update_pio=True,
               convert_to_cdf5=False, delta_ssh_threshold=None):
    """
    Adjust the sea surface height or land-ice pressure to be dynamically
    consistent with one another.  A series of short model runs are performed,
    each with

    Parameters
    ----------
    variable : {'ssh', 'landIcePressure'}
        The variable to adjust

    iteration_count : int
        The number of iterations of adjustment

    step : compass.Step
        the step for performing SSH or land-ice pressure adjustment

    update_pio : bool, optional
        Whether to update PIO tasks and stride

    convert_to_cdf5 : bool, optional
        Whether to convert files to CDF5 format with ncks after writing them
        out.  This is intended to improve MPAS-Ocean performance, since reading
        in NETCDF4 format files can be very slow.
    """
    ntasks = step.ntasks
    config = step.config
    logger = step.logger
    out_filename = None

    if variable not in ['ssh', 'landIcePressure']:
        raise ValueError(f"Unknown variable to modify: {variable}")

    if update_pio:
        step.update_namelist_pio('namelist.ocean')
    partition(ntasks, config, logger)

    for iterIndex in range(iteration_count):
        logger.info(f" * Iteration {iterIndex + 1}/{iteration_count}")

        in_filename = f'adjusting_init{iterIndex}.nc'
        out_filename = f'adjusting_init{iterIndex + 1}.nc'
        symlink(in_filename, 'adjusting_init.nc')

        logger.info("   * Running forward model")
        run_model(step, update_pio=False, partition_graph=False)
        logger.info("   - Complete")

        logger.info("   * Updating SSH or land-ice pressure")

        with xarray.open_dataset(in_filename) as ds:

            # keep the data set with Time for output
            ds_out = ds

            ds = ds.isel(Time=0)

            on_a_sphere = ds.attrs['on_a_sphere'].lower() == 'yes'

            initSSH = ds.ssh
            if 'minLevelCell' in ds:
                minLevelCell = ds.minLevelCell - 1
            else:
                minLevelCell = xarray.zeros_like(ds.maxLevelCell)

            with xarray.open_dataset('output_ssh.nc') as ds_ssh:
                # get the last time entry
                ds_ssh = ds_ssh.isel(Time=ds_ssh.sizes['Time'] - 1)
                finalSSH = ds_ssh.ssh
                topDensity = ds_ssh.density.isel(nVertLevels=minLevelCell)

            mask = numpy.logical_and(ds.maxLevelCell > 0,
                                     ds.modifyLandIcePressureMask == 1)

            deltaSSH = mask * (finalSSH - initSSH)

            # then, modify the SSH or land-ice pressure
            if variable == 'ssh':
                ssh = finalSSH.expand_dims(dim='Time', axis=0)
                ds_out['ssh'] = ssh
                # also update the landIceDraft variable, which will be used to
                # compensate for the SSH due to land-ice pressure when
                # computing sea-surface tilt
                ds_out['landIceDraft'] = ssh
                # we also need to stretch layerThickness to be compatible with
                # the new SSH
                stretch = ((finalSSH + ds.bottomDepth) /
                           (initSSH + ds.bottomDepth))
                ds_out['layerThickness'] = ds_out.layerThickness * stretch
                landIcePressure = ds.landIcePressure.values
            else:
                # Moving the SSH up or down by deltaSSH would change the
                # land-ice pressure by density(SSH)*g*deltaSSH. If deltaSSH is
                # positive (moving up), it means the land-ice pressure is too
                # small and if deltaSSH is negative (moving down), it means
                # land-ice pressure is too large, the sign of the second term
                # makes sense.
                gravity = constants['SHR_CONST_G']
                deltaLandIcePressure = topDensity * gravity * deltaSSH

                landIcePressure = numpy.maximum(
                    0.0, ds.landIcePressure + deltaLandIcePressure)

                ds_out['landIcePressure'] = \
                    landIcePressure.expand_dims(dim='Time', axis=0)

                finalSSH = initSSH

            if convert_to_cdf5:
                name, ext = os.path.splitext(out_filename)
                write_filename = f'{name}_before_cdf5{ext}'
            else:
                write_filename = out_filename
            write_netcdf(ds_out, write_filename)
            if convert_to_cdf5:
                args = ['ncks', '-O', '-5', write_filename, out_filename]
                subprocess.check_call(args)

            # Write the largest change in SSH and its lon/lat to a file
            with open(f'maxDeltaSSH_{iterIndex:03d}.log', 'w') as \
                    log_file:

                mask = landIcePressure > 0.
                iCell = numpy.abs(deltaSSH.where(mask)).argmax().values

                ds_cell = ds.isel(nCells=iCell)
                deltaSSHMax = deltaSSH.isel(nCells=iCell).values

                if on_a_sphere:
                    coords = (f'lon/lat: '
                              f'{numpy.rad2deg(ds_cell.lonCell.values):f} '
                              f'{numpy.rad2deg(ds_cell.latCell.values):f}')
                else:
                    coords = (f'x/y: {1e-3 * ds_cell.xCell.values:f} '
                              f'{1e-3 * ds_cell.yCell.values:f}')
                string = (f'deltaSSHMax: '
                          f'{deltaSSHMax:g}, {coords}')
                logger.info(f'     {string}')
                log_file.write(f'{string}\n')
                string = (f'ssh: {finalSSH.isel(nCells=iCell).values:g}, '
                          f'landIcePressure: '
                          f'{landIcePressure.isel(nCells=iCell).values:g}')
                logger.info(f'     {string}')
                log_file.write(f'{string}\n')

                if delta_ssh_threshold is not None:
                    if abs(deltaSSHMax) < delta_ssh_threshold:
                        break

        logger.info("   - Complete\n")

    if out_filename is not None:
        shutil.copy(out_filename, 'adjusted_init.nc')
