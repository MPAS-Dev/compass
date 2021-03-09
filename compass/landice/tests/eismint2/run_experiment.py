import numpy
from netCDF4 import Dataset as NetCDFFile
import shutil

from mpas_tools.logging import check_call

from compass.io import add_input_file, add_output_file
from compass.namelist import add_namelist_file, generate_namelist
from compass.streams import add_streams_file, generate_streams
from compass.model import add_model_as_input, run_model


def collect(testcase, step):
    """
    Update the dictionary of step properties

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case, which should not be
        modified here

    step : dict
        A dictionary of properties of this step, which can be updated
    """
    defaults = dict(max_memory=1000, max_disk=1000, threads=1,
                    suffixes=['landice'])
    for key, value in defaults.items():
        step.setdefault(key, value)

    step.setdefault('min_cores', step['cores'])

    experiment = step['experiment']

    # most runs will just have a namelist.landice and a streams.landice but
    # the restart_run step of the restart_test runs the model twice, the second
    # time with namelist.landice.rst and streams.landice.rst
    for suffix in step['suffixes']:
        add_namelist_file(
            step, 'compass.landice.tests.eismint2', 'namelist.landice',
            out_name='namelist.{}'.format(suffix))

        add_streams_file(
            step, 'compass.landice.tests.eismint2', 'streams.landice',
            out_name='streams.{}'.format(suffix))

    if experiment in ('a', 'f', 'g'):
        add_input_file(step, filename='landice_grid.nc',
                       target='../setup_mesh/landice_grid.nc')
    else:
        add_input_file(step, filename='experiment_a_output.nc',
                       target='../experiment_a/output.nc')

    add_input_file(step, filename='graph.info',
                   target='../setup_mesh/graph.info')

    add_output_file(step, filename='output.nc')


def setup(step, config):
    """
    Set up the test case in the work directory, including downloading any
    dependencies

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step from the ``collect()`` function

    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core, configuration and test case
    """
    # again, most runs will just have a namelist.landice and a streams.landice
    # but the restart_run step of the restart_test runs the model twice, the
    # second time with namelist.landice.rst and streams.landice.rst
    for suffix in step['suffixes']:
        generate_namelist(step, config, out_name='namelist.{}'.format(suffix))
        generate_streams(step, config, out_name='streams.{}'.format(suffix))

    add_model_as_input(step, config)


def run(step, test_suite, config, logger):
    """
    Run this step of the test case

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step from the ``collect()``
        function, with modifications from the ``setup()`` function.

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step
    """

    experiment = step['experiment']

    _setup_eismint2_initial_conditions(config, logger, experiment,
                                       filename='initial_condition.nc')

    # again, most runs will just have a namelist.landice and a streams.landice
    # but the restart_run step of the restart_test runs the model twice, the
    # second time with namelist.landice.rst and streams.landice.rst
    for suffix in step['suffixes']:
        run_model(step, config, logger, namelist='namelist.{}'.format(suffix),
                  streams='streams.{}'.format(suffix))


def _setup_eismint2_initial_conditions(config, logger, experiment, filename):
    """
    Add the initial condition for the given EISMINT2 experiment to the given
    MPAS mesh file

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step

    experiment : {'a', 'b', 'c', 'd', 'f', 'g'}
        The name of the experiment

    filename : str
        file to add the initial condition to

    """
    section = config['eismint2']

    if experiment in ('a', 'b', 'c', 'd', 'f', 'g'):
        logger.info('Setting up EISMINT2 Experiment {}'.format(experiment))
    else:
        raise ValueError("Invalid experiment specified: {}.  Please specify "
                         "an experiment between 'a' and 'g', excluding "
                         "'e'".format(experiment))

    # Setup dictionaries of parameter values for each experiment
    # Mmax: Maximum SMB at center of domain (m a-1)
    # Sb: gradient of SMB with horizontal distance (m a-1 km-1)
    # Rel: radial distance from summit where SMB = 0 (km)
    # Tmin: surface temperature at summit (K)
    # ST: gradient of air temperature with horizontal distance (K km-1)
    # beta: basal traction coefficient (Pa m-1 a)
    #       Note: beta is the inverse of parameter B in Payne et al. (2000)
    exp_params = {'a': {'Mmax': 0.5, 'Sb': 10.0**-2, 'Rel': 450.0,
                        'Tmin': 238.15, 'ST': 1.67e-2, 'beta': 1.0e8},
                  'b': {'Mmax': 0.5, 'Sb': 10.0**-2, 'Rel': 450.0,
                        'Tmin': 243.15, 'ST': 1.67e-2, 'beta': 1.0e8},
                  'c': {'Mmax': 0.25, 'Sb': 10.0**-2, 'Rel': 425.0,
                        'Tmin': 238.15, 'ST': 1.67e-2, 'beta': 1.0e8},
                  'd': {'Mmax': 0.5, 'Sb': 10.0**-2, 'Rel': 425.0,
                        'Tmin': 238.15, 'ST': 1.67e-2, 'beta': 1.0e8},
                  'f': {'Mmax': 0.5, 'Sb': 10.0**-2, 'Rel': 450.0,
                        'Tmin': 223.15, 'ST': 1.67e-2, 'beta': 1.0e8},
                  'g': {'Mmax': 0.5, 'Sb': 10.0**-2, 'Rel': 450.0,
                        'Tmin': 238.15, 'ST': 1.67e-2, 'beta': 1.0e3}}
    xsummit = 750000.0
    ysummit = 750000.0
    rhoi = 910.0
    scyr = 3600.0 * 24.0 * 365.0

    # Some experiments start from scratch, others start from the SS of a previous experiment
    if experiment in ('a', 'f', 'g'):
        # we will build the mesh from scratch
        shutil.copyfile('landice_grid.nc', filename)
    else:
        # use the final state of experiment A
        args = ['ncks', '-O', '-d', 'Time,-1', 'experiment_a_output.nc',
                filename]
        check_call(args, logger)

    # Open the new input file, get needed dimensions & variables
    gridfile = NetCDFFile(filename, 'r+')
    nVertLevels = len(gridfile.dimensions['nVertLevels'])
    # Get variables
    xCell = gridfile.variables['xCell'][:]
    yCell = gridfile.variables['yCell'][:]
    xEdge = gridfile.variables['xEdge'][:]
    yEdge = gridfile.variables['yEdge'][:]
    xVertex = gridfile.variables['xVertex'][:]
    yVertex = gridfile.variables['yVertex'][:]

    # ===================
    # initial conditions
    # ===================
    # If starting from scratch, setup dimension variables and initial condition
    # variables
    if experiment in ('a', 'f', 'g'):
        # Find center of domain
        x0 = xCell[:].min() + 0.5 * (xCell[:].max() - xCell[:].min())
        y0 = yCell[:].min() + 0.5 * (yCell[:].max() - yCell[:].min())
        # Calculate distance of each cell center from dome center
        r = ((xCell[:] - x0)**2 + (yCell[:] - y0)**2)**0.5

        # Center the dome in the center of the cell that is closest to the
        # center of the domain.
        centerCellIndex = numpy.abs(r[:]).argmin()
        # EISMINT-2 puts the center of the domain at 750,750 km instead of 0,0.
        # Adjust to use that origin.

        xShift = -1.0 * xCell[centerCellIndex] + xsummit
        yShift = -1.0 * yCell[centerCellIndex] + ysummit
        xCell[:] = xCell[:] + xShift
        yCell[:] = yCell[:] + yShift
        xEdge[:] = xEdge[:] + xShift
        yEdge[:] = yEdge[:] + yShift
        xVertex[:] = xVertex[:] + xShift
        yVertex[:] = yVertex[:] + yShift
        gridfile.variables['xCell'][:] = xCell[:]
        gridfile.variables['yCell'][:] = yCell[:]
        gridfile.variables['xEdge'][:] = xEdge[:]
        gridfile.variables['yEdge'][:] = yEdge[:]
        gridfile.variables['xVertex'][:] = xVertex[:]
        gridfile.variables['yVertex'][:] = yVertex[:]

        # Assign initial condition variable values for EISMINT-2 experiment
        # Start with no ice
        gridfile.variables['thickness'][:] = 0.0
        # flat bed at sea level
        gridfile.variables['bedTopography'][:] = 0.0
        # constant, arbitrary temperature, degrees K (doesn't matter since
        # there is no ice initially)
        gridfile.variables['temperature'][:] = 273.15
        # Setup layerThicknessFractions
        gridfile.variables['layerThicknessFractions'][:] = 1.0 / nVertLevels
    else:
        StrLen = len(gridfile.dimensions['StrLen'])
        gridfile.variables['xtime'][0, :] = list(
            '000000-01-01_00:00:00'.ljust(StrLen, ' '))

    # Now update/set origin location and distance array
    r = ((xCell[:] - xsummit)**2 + (yCell[:] - ysummit)**2)**0.5

    # ===================
    # boundary conditions
    # ===================
    # Define values prescribed by Payne et al. 2000 paper.

    params = exp_params[experiment]
    logger.info("Parameters for this experiment: {}".format(params))

    # SMB field specified by EISMINT, constant in time for EISMINT2
    # It is a function of geographical position (not elevation)

    # maximum accumulation rate [m/yr] converted to [m/s]
    Mmax = params['Mmax'] / scyr
    # gradient of accumulation rate change with horizontal distance  [m/a/km]
    # converted to [m/s/m]
    Sb = params['Sb'] / scyr / 1000.0
    # accumulation rate at 0 position  [km] converted to [m]
    Rel = params['Rel'] * 1000.0

    SMB = numpy.minimum(Mmax, Sb * (Rel - r))  # [m ice/s]
    SMB = SMB * rhoi  # in kg/m2/s
    if 'sfcMassBal' in gridfile.variables:
        sfcMassBalVar = gridfile.variables['sfcMassBal']
    else:
        datatype = gridfile.variables[
            'xCell'].dtype  # Get the datatype for double precision float
        sfcMassBalVar = gridfile.createVariable('sfcMassBal', datatype,
                                                ('Time', 'nCells'))
    sfcMassBalVar[0, :] = SMB

    # Surface temperature

    # minimum surface air temperature [K]
    Tmin = params['Tmin']
    # gradient of air temperature change with horizontal distance [K/km]
    # converted to [K/m]
    ST = params['ST'] / 1000.0

    if 'surfaceAirTemperature' in gridfile.variables:
        surfaceAirTemperatureVar = gridfile.variables['surfaceAirTemperature']
    else:
        datatype = gridfile.variables[
            'xCell'].dtype  # Get the datatype for double precision float
        surfaceAirTemperatureVar = gridfile.createVariable(
            'surfaceAirTemperature', datatype, ('Time', 'nCells'))
    surfaceAirTemperatureVar[0, :] = Tmin + ST * r

    # beta
    beta = params['beta']
    if 'beta' in gridfile.variables:
        betaVar = gridfile.variables['beta']
    else:
        datatype = gridfile.variables[
            'xCell'].dtype  # Get the datatype for double precision float
        betaVar = gridfile.createVariable('beta', datatype, ('Time', 'nCells'))
    betaVar[0, :] = beta

    gridfile.close()
    logger.info('Successfully added initial conditions for EISMINT2, '
                'experiment {} to the file: {}'.format(experiment, filename))
