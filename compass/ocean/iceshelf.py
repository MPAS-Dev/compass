import numpy
from netCDF4 import Dataset
import shutil

from mpas_tools.cime.constants import constants
from compass.io import symlink
from compass.model import partition, run_model


def compute_land_ice_pressure_and_draft(ssh, modifySSHMask, ref_density):
    """
    Compute the pressure from and overlying ice shelf and the ice-shelf draft

    Parameters
    ----------
    ssh : xarray.DataArray
        The sea surface height (the ice draft)

    modifySSHMask : xarray.DataArray
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
        modifySSHMask*numpy.maximum(-ref_density * gravity * ssh, 0.)
    landIceDraft = ssh
    return landIcePressure, landIceDraft


def adjust_ssh(variable, iteration_count, config, cores, logger):
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

    config : configparser.ConfigParser
        Configuration options for this testcase

    cores : int
        The number of cores to use in this step of the testcase

    logger : logging.Logger
        A logger for output from the step
    """

    if variable not in ['ssh', 'landIcePressure']:
        raise ValueError("Unknown variable to modify: {}".format(variable))

    partition(cores, logger)

    for iterIndex in range(iteration_count):
        logger.info(" * Iteration {}/{}".format(iterIndex + 1,
                                                iteration_count))

        symlink('adjusting_init{}.nc'.format(iterIndex), 'adjusting_init.nc')

        logger.info("   * Running forward model")
        run_model(config, core='ocean', core_count=cores, logger=logger,
                  threads=1)
        logger.info("   - Complete")

        logger.info("   * Updating SSH or land-ice pressure")

        # copy the init file first
        shutil.copy('adjusting_init{}.nc'.format(iterIndex),
                    'adjusting_init{}.nc'.format(iterIndex+1))

        symlink('adjusting_init{}.nc'.format(iterIndex+1),
                'adjusting_init.nc')

        with Dataset('adjusting_init.nc', 'r+') as ds:

            on_a_sphere = ds.on_a_sphere.lower() == 'yes'

            nVertLevels = len(ds.dimensions['nVertLevels'])
            initSSH = ds.variables['ssh'][0, :]
            bottomDepth = ds.variables['bottomDepth'][:]
            modifySSHMask = ds.variables['modifySSHMask'][0, :]
            landIcePressure = ds.variables['landIcePressure'][0, :]
            lonCell = ds.variables['lonCell'][:]
            latCell = ds.variables['latCell'][:]
            xCell = ds.variables['xCell'][:]
            yCell = ds.variables['yCell'][:]
            maxLevelCell = ds.variables['maxLevelCell'][:]

            with Dataset('output_ssh.nc', 'r') as ds_ssh:
                nTime = len(ds_ssh.dimensions['Time'])
                finalSSH = ds_ssh.variables['ssh'][nTime - 1, :]
                topDensity = ds_ssh.variables['density'][nTime - 1, :, 0]

            mask = numpy.logical_and(maxLevelCell > 0, modifySSHMask == 1)

            deltaSSH = mask * (finalSSH - initSSH)

            # then, modify the SSH or land-ice pressure
            if variable == 'ssh':
                ds.variables['ssh'][0, :] = finalSSH
                # also update the landIceDraft variable, which will be used to
                # compensate for the SSH due to land-ice pressure when
                # computing sea-surface tilt
                ds.variables['landIceDraft'][0, :] = finalSSH
                # we also need to stretch layerThickness to be compatible with
                # the new SSH
                stretch = (finalSSH + bottomDepth) / (initSSH + bottomDepth)
                layerThickness = ds.variables['layerThickness']
                for k in range(nVertLevels):
                    layerThickness[0, :, k] *= stretch
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
                    0.0, landIcePressure + deltaLandIcePressure)

                ds.variables['landIcePressure'][0, :] = landIcePressure

                finalSSH = initSSH

        # Write the largest change in SSH and its lon/lat to a file
        with open('maxDeltaSSH_{:03d}.log'.format(iterIndex), 'w') as log_file:

            indices = numpy.nonzero(landIcePressure)[0]
            index = numpy.argmax(numpy.abs(deltaSSH[indices]))
            iCell = indices[index]
            if on_a_sphere:
                coords = 'lon/lat: {:f} {:f}'.format(
                    numpy.rad2deg(lonCell[iCell]),
                    numpy.rad2deg(latCell[iCell]))
            else:
                coords = 'x/y: {:f} {:f}'.format(1e-3 * xCell[iCell],
                                                 1e-3 * yCell[iCell])
            string = 'deltaSSHMax: {:g}, {}'.format(
                deltaSSH[iCell], coords)
            logger.info('     {}'.format(string))
            log_file.write('{}\n'.format(string))
            string = 'ssh: {:g}, landIcePressure: {:g}'.format(
                finalSSH[iCell], landIcePressure[iCell])
            logger.info('     {}'.format(string))
            log_file.write('{}\n'.format(string))

        logger.info("   - Complete\n")

    shutil.copy('adjusting_init.nc', 'adjusted_init.nc')
