import os

from compass.testcase import get_step_default
from compass.io import symlink, add_input_file, add_output_file
from compass.namelist import add_namelist_file, add_namelist_options,\
    generate_namelist
from compass.streams import add_streams_file, generate_streams
from compass.model import run_model


def collect(resolution, cores, min_cores=None, max_memory=1000,
            max_disk=1000, threads=1, nu=None):
    """
    Get a dictionary of step properties

    Parameters
    ----------
    resolution : {'1km', '4km', '10km'}
        The name of the resolution to run at

    cores : int
        The number of cores to run on in forward runs. If this many cores are
        available on the machine or batch job, the task will run on that
        number. If fewer are available (but no fewer than min_cores), the job
        will run on all available cores instead.

    min_cores : int, optional
        The minimum allowed cores.  If that number of cores are not available
        on the machine or in the batch job, the run will fail.  By default,
        ``min_cores = cores``

    max_memory : int, optional
        The maximum amount of memory (in MB) this step is allowed to use

    max_disk : int, optional
        The maximum amount of disk space  (in MB) this step is allowed to use

    threads : int, optional
        The number of threads to run with during forward runs

    nu : float, optional
        The viscosity for this step

    Returns
    -------
    step : dict
        A dictionary of properties of this step
    """
    step = get_step_default(__name__)
    step['resolution'] = resolution
    step['cores'] = cores
    step['max_memory'] = max_memory
    step['max_disk'] = max_disk
    if min_cores is None:
        min_cores = cores
    step['min_cores'] = min_cores
    step['threads'] = threads

    add_namelist_file(step, 'compass.ocean.tests.baroclinic_channel',
                      'namelist.forward')
    add_namelist_file(step, 'compass.ocean.tests.baroclinic_channel',
                      'namelist.{}.forward'.format(resolution))
    if nu is not None:
        # update the viscosity to the requested value
        add_namelist_options(step, {'config_mom_del2': '{}'.format(nu)})

    add_streams_file(step, 'compass.ocean.tests.baroclinic_channel',
                     'streams.forward')

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
        Configuration options for this testcase, a combination of the defaults
        for the machine, core, configuration and testcase
    """
    step_dir = step['work_dir']

    # generate the namelist and streams files file from the various files and
    # replacements we have collected
    generate_namelist(step, config)
    generate_streams(step, config)

    # make a link to the ocean_model executable
    symlink(os.path.abspath(config.get('executables', 'model')),
            os.path.join(step_dir, 'ocean_model'))

    add_input_file(step, filename='init.nc',
                   target='../initial_state/ocean.nc')
    add_input_file(step, filename='graph.info',
                   target='../initial_state/culled_graph.info')

    add_output_file(step, filename='output.nc')


def run(step, test_suite, config, logger):
    """
    Run this step of the testcase

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step from the ``collect()``
        function, with modifications from the ``setup()`` function.

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this testcase, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step
    """
    run_model(step, config, logger)
