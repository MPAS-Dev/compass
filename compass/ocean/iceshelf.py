import os
import shutil
import subprocess

import numpy
import xarray
from mpas_tools.cime.constants import constants
from mpas_tools.io import write_netcdf

from compass.io import symlink
from compass.model import partition, run_model


def compute_land_ice_pressure_from_thickness(land_ice_thickness, modify_mask,
                                             land_ice_density=None):
    """
    Compute the pressure from and overlying ice shelf

    Parameters
    ----------
    land_ice_thickness: xarray.DataArray
        The ice thickness

    modify_mask : xarray.DataArray
        A mask that is 1 where ``landIcePressure`` can be deviate from 0

    land_ice_density : float, optional
        A reference density for land ice

    Returns
    -------
    land_ice_pressure : xarray.DataArray
        The pressure from the overlying land ice on the ocean
    """
    gravity = constants['SHR_CONST_G']
    if land_ice_density is None:
        land_ice_density = constants['SHR_CONST_RHOICE']
    land_ice_pressure = modify_mask * \
        numpy.maximum(land_ice_density * gravity * land_ice_thickness, 0.)
    return land_ice_pressure


def compute_land_ice_density_from_draft(land_ice_draft, land_ice_thickness,
                                        floating_mask, ref_density=None):
    """
    Compute the spatially-averaged ice density needed to match the ice draft

    Parameters
    ----------
    land_ice_draft : xarray.DataArray
        The ice draft (sea surface height)

    land_ice_thickness: xarray.DataArray
        The ice thickness

    floating_mask : xarray.DataArray
        A mask that is 1 where the ice is assumed in hydrostatic equilibrium

    ref_density : float, optional
        A reference density for seawater displaced by the ice shelf

    Returns
    -------
    land_ice_density: float
        The ice density
    """
    if ref_density is None:
        ref_density = constants['SHR_CONST_RHOSW']
    land_ice_draft = numpy.where(floating_mask, land_ice_draft, numpy.nan)
    land_ice_thickness = numpy.where(floating_mask, land_ice_thickness,
                                     numpy.nan)
    land_ice_density = \
        numpy.nanmean(-ref_density * land_ice_draft / land_ice_thickness)
    return land_ice_density


def compute_land_ice_pressure_from_draft(land_ice_draft, modify_mask,
                                         ref_density=None):
    """
    Compute the pressure from and overlying ice shelf

    Parameters
    ----------
    land_ice_draft : xarray.DataArray
        The ice draft (sea surface height)

    modify_mask : xarray.DataArray
        A mask that is 1 where ``landIcePressure`` can be deviate from 0

    ref_density : float, optional
        A reference density for seawater displaced by the ice shelf

    Returns
    -------
    land_ice_pressure : xarray.DataArray
        The pressure from the overlying land ice on the ocean
    """
    gravity = constants['SHR_CONST_G']
    if ref_density is None:
        ref_density = constants['SHR_CONST_RHOSW']
    land_ice_pressure = \
        modify_mask * numpy.maximum(-ref_density * gravity * land_ice_draft,
                                    0.)
    return land_ice_pressure


def compute_land_ice_draft_from_pressure(land_ice_pressure, modify_mask,
                                         ref_density=None):
    """
    Compute the ice-shelf draft associated with the pressure from an overlying
    ice shelf

    Parameters
    ----------
    land_ice_pressure : xarray.DataArray
        The pressure from the overlying land ice on the ocean

    modify_mask : xarray.DataArray
        A mask that is 1 where ``landIcePressure`` can be deviate from 0

    ref_density : float, optional
        A reference density for seawater displaced by the ice shelf

    Returns
    -------
    land_ice_draft : xarray.DataArray
        The ice draft
    """
    gravity = constants['SHR_CONST_G']
    if ref_density is None:
        ref_density = constants['SHR_CONST_RHOSW']
    land_ice_draft = \
        - (modify_mask * land_ice_pressure / (ref_density * gravity))
    return land_ice_draft


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

    with xarray.open_dataset('adjusting_init0.nc') as ds:
        ds = ds.isel(Time=0)
        orig_ssh = ds.ssh
        orig_land_ice_pressure = ds.landIcePressure

        on_a_sphere = ds.attrs['on_a_sphere'].lower() == 'yes'

        modify_mask = numpy.logical_and(ds.maxLevelCell > 0,
                                        ds.sshAdjustmentMask == 1)

    for iter_index in range(iteration_count):
        logger.info(f" * Iteration {iter_index + 1}/{iteration_count}")

        in_filename = f'adjusting_init{iter_index}.nc'
        out_filename = f'adjusting_init{iter_index + 1}.nc'
        symlink(in_filename, 'adjusting_init.nc')

        logger.info("   * Running forward model")
        run_model(step, update_pio=False, partition_graph=False)
        logger.info("   - Complete")

        logger.info("   * Updating SSH or land-ice pressure")

        with xarray.open_dataset(in_filename) as ds:

            # keep the data set with Time for output
            ds_out = ds

            ds = ds.isel(Time=0)

            init_ssh = ds.ssh
            if 'minLevelCell' in ds:
                minLevelCell = ds.minLevelCell - 1
            else:
                minLevelCell = xarray.zeros_like(ds.maxLevelCell)

            with xarray.open_dataset('output_ssh.nc') as ds_ssh:
                # get the last time entry
                ds_ssh = ds_ssh.isel(Time=-1)
                final_ssh = ds_ssh.ssh
                topDensity = ds_ssh.density.isel(nVertLevels=minLevelCell)

            delta_ssh = modify_mask * (final_ssh - init_ssh)

            # then, modify the SSH or land-ice pressure
            if variable == 'ssh':
                ssh = final_ssh.expand_dims(dim='Time', axis=0)
                ds_out['ssh'] = ssh
                # also update the landIceDraft variable, which will be used to
                # compensate for the SSH due to land-ice pressure when
                # computing sea-surface tilt
                ds_out['landIceDraft'] = ssh
                # we also need to stretch layerThickness to be compatible with
                # the new SSH
                stretch = ((final_ssh + ds.bottomDepth) /
                           (init_ssh + ds.bottomDepth))
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
                deltaLandIcePressure = topDensity * gravity * delta_ssh

                landIcePressure = numpy.maximum(
                    0.0, ds.landIcePressure + deltaLandIcePressure)

                ds_out['landIcePressure'] = \
                    landIcePressure.expand_dims(dim='Time', axis=0)

                final_ssh = init_ssh

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
            with open(f'maxDeltaSSH_{iter_index:03d}.log', 'w') as \
                    log_file:

                mask = landIcePressure > 0.
                icell = numpy.abs(delta_ssh.where(mask)).argmax().values

                ds_cell = ds.isel(nCells=icell)
                delta_ssh_max = delta_ssh.isel(nCells=icell).values

                if on_a_sphere:
                    coords = (f'lon/lat: '
                              f'{numpy.rad2deg(ds_cell.lonCell.values):f} '
                              f'{numpy.rad2deg(ds_cell.latCell.values):f}')
                else:
                    coords = (f'x/y: {1e-3 * ds_cell.xCell.values:f} '
                              f'{1e-3 * ds_cell.yCell.values:f}')
                string = (f'delta_ssh_max: '
                          f'{delta_ssh_max:g}, {coords}')
                logger.info(f'     {string}')
                log_file.write(f'{string}\n')
                string = (f'ssh: {final_ssh.isel(nCells=icell).values:g}, '
                          f'landIcePressure: '
                          f'{landIcePressure.isel(nCells=icell).values:g}')
                logger.info(f'     {string}')
                log_file.write(f'{string}\n')

                if delta_ssh_threshold is not None:
                    if abs(delta_ssh_max) < delta_ssh_threshold:
                        break

        logger.info("   - Complete\n")

    if out_filename is not None:
        shutil.copy(out_filename, 'adjusted_init.nc')

        with xarray.open_dataset('adjusted_init.nc') as ds:
            ds = ds.isel(Time=0)
            final_ssh = ds.ssh
            final_land_ice_pressure = ds.landIcePressure
            delta_ssh = final_ssh - orig_ssh
            masked_delta_ssh = modify_mask * delta_ssh
            delta_land_ice_pressure = \
                final_land_ice_pressure - orig_land_ice_pressure
            masked_delta_land_ice_pressure = \
                modify_mask * delta_land_ice_pressure
            ds_out = xarray.Dataset()
            ds_out['delta_ssh'] = delta_ssh
            ds_out['masked_delta_ssh'] = masked_delta_ssh
            ds_out['delta_land_ice_pressure'] = delta_land_ice_pressure
            ds_out['masked_delta_land_ice_pressure'] = \
                masked_delta_land_ice_pressure
            write_netcdf(ds_out, 'total_delta.nc')
