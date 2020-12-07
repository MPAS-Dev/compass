import xarray
import os

from mpas_tools.io import write_netcdf

from compass.testcase import get_step_default
from compass.io import download, symlink


# "resolution" is just an example argument.  The argument can be any parameter
# that distinguishes different variants of a test
def collect(resolution):
    """
    Get a dictionary of step properties

    Parameters
    ----------
    resolution : {'1km', '2km'}
        The name of the resolution to run at

    Returns
    -------
    step : dict
        A dictionary of properties of this step
    """
    # get some default information for the step
    step = get_step_default(__name__)
    # add any parameters or other information you would like to have when you
    # are setting up or running the testcase or its steps
    step['resolution'] = resolution

    return step


def setup(step, config):
    """
    Set up the test case in the work directory, including downloading any
    dependencies

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step from the ``collect()`` function

    config : configparser.ConfigParser
        Configuration options for this step, a combination of the defaults for
        the machine, core, configuration and testcase
    """
    resolution = step['resolution']
    testcase = step['testcase']
    # This is a way to handle a few parameters that are specific to different
    # testcases or resolutions, all of which can be handled by this function
    res_params = {'1km': {'parameter4': 1.0,
                          'parameter5': 500},
                  '2km': {'parameter4': 2.0,
                          'parameter5': 250}}

    test_params = {'test1': {'filename': 'particle_regions.151113.nc'},
                   'test2': {'filename': 'layer_depth.80Layer.180619.nc'}}

    # copy the appropriate parameters into the step dict for use in run
    if resolution not in res_params:
        raise ValueError('Unsupported resolution {}. Supported values are: '
                         '{}'.format(resolution, list(res_params)))
    res_params = res_params[resolution]

    # add the parameters for this resolution to the step dictionary so they
    # are available to the run() function
    for param in res_params:
        step[param] = res_params[param]

    if testcase not in test_params:
        raise ValueError('Unsupported testcase name {}. Supported testcases '
                         'are: {}'.format(testcase, list(test_params)))
    test_params = test_params[testcase]

    # add the parameters for this testcase to the step dictionary so they
    # are available to the run() function
    for param in test_params:
        step[param] = test_params[param]

    initial_condition_database = config.get('paths',
                                            'initial_condition_database')
    step_dir = step['work_dir']

    # one of the required parts of setup is to define any input files from
    # other steps or testcases that are required by this step, and any output
    # files that are produced by this step that might be used in other steps
    # or testcases.  This allows COMPASS to determine dependencies between
    # testcases and their steps
    inputs = []
    outputs = []

    # download an input file if it's not already in the initial condition
    # database
    filename = download(
        file_name=step['filename'],
        url='https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/'
            'mpas-ocean/initial_condition_database',
        config=config, dest_path=initial_condition_database)

    inputs.append(filename)

    symlink(filename, os.path.join(step_dir, 'input_file.nc'))

    # list all the output files that will be produced in the step1 subdirectory
    for file in ['output_file.nc']:
        outputs.append(os.path.join(step_dir, file))

    step['inputs'] = inputs
    step['outputs'] = outputs


def run(step, test_suite, config, logger):
    """
    Run this step of the testcase

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step from the ``collect()`` function,
        with modifications from the ``setup()`` function.

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this testcase, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step
    """
    test_config = config['example_compact']
    parameter1 = test_config.getfloat('parameter1')
    parameter2 = test_config.getboolean('parameter2')
    testcase = step['testcase']

    ds = xarray.open_dataset('input_file.nc')
    write_netcdf(ds, 'output_file.nc')
