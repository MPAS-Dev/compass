import xarray
import os

from mpas_tools.io import write_netcdf

from compass.testcase import get_step_default
from compass.io import symlink


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
    step_dir = step['work_dir']

    # one of the required parts of setup is to define any input files from
    # other steps or testcases that are required by this step, and any output
    # files that are produced by this step that might be used in other steps
    # or testcases.  This allows COMPASS to determine dependencies between
    # testcases and their steps
    inputs = []
    outputs = []

    # add a link to the output from step1 and add it as an input to this step
    filename = os.path.abspath(os.path.join(step_dir,
                                            '../step1/output_file.nc'))
    symlink(filename, os.path.join(step_dir, 'input_file.nc'))
    inputs.append(filename)

    # list all the output files that will be produced in the step1 subdirectory
    for file in ['output_file.nc']:
        outputs.append(os.path.join(step_dir, file))

    step['inputs'] = inputs
    step['outputs'] = outputs


def run(step, config):
    """
    Run this step of the testcase

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step from the ``collect()`` function,
        with modifications from the ``setup()`` function.

    config : configparser.ConfigParser
        Configuration options for this testcase, a combination of the defaults
        for the machine, core and configuration
    """
    ds = xarray.open_dataset('input_file.nc')
    write_netcdf(ds, 'output_file.nc')
