import os
from netCDF4 import Dataset

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
    defaults = dict(max_memory=1000, max_disk=1000, threads=1)
    for key, value in defaults.items():
        step.setdefault(key, value)

    step.setdefault('min_cores', step['cores'])

    add_namelist_file(
        step, 'compass.landice.tests.enthalpy_benchmark',
        'namelist.landice')
    add_streams_file(
        step, 'compass.landice.tests.enthalpy_benchmark',
        'streams.landice')

    add_input_file(step, filename='landice_grid.nc',
                   target='../setup_mesh/landice_grid.nc')
    add_input_file(step, filename='graph.info',
                   target='../setup_mesh/graph.info')

    if 'restart_filename' in step:
        target = step['restart_filename']
        filename = os.path.basename(target)
        add_input_file(step, filename=filename, target=target)

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
    generate_namelist(step, config)
    generate_streams(step, config)

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
    section = config['enthalpy_benchmark']
    if 'restart_filename' in step:
        _update_surface_air_temperature(step, section)

    run_model(step, config, logger)


def _update_surface_air_temperature(step, section):
    phase = step['name']
    # set the surface air temperature
    option = '{}_surface_air_temperature'.format(phase)
    surface_air_temperature = section.getfloat(option)
    filename = step['restart_filename']
    with Dataset(filename, 'r+') as data:
        data.variables['surfaceAirTemperature'][0, :] = \
            surface_air_temperature
