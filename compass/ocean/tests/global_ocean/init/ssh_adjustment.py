import os

from compass.testcase import get_step_default
from compass.io import symlink
from compass import namelist, streams
from compass.parallel import update_namelist_pio
from compass.ocean.iceshelf import adjust_ssh
from compass.ocean.tests.global_ocean.subdir import get_mesh_relative_path, \
    get_initial_condition_relative_path


def collect(mesh_name, cores, min_cores=None, max_memory=1000,
            max_disk=1000, testcase_module=None,
            namelist_file=None, streams_file=None):
    """
    Get a dictionary of step properties

    Parameters
    ----------
    mesh_name : str
        The name of the mesh

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

    testcase_module : str, optional
        The module for the testcase

    namelist_file : str, optional
        The name of a namelist file in the testcase package directory

    streams_file : str, optional
        The name of a streams file in the testcase package directory

    Returns
    -------
    step : dict
        A dictionary of properties of this step
    """
    step = get_step_default(__name__)
    step['mesh_name'] = mesh_name
    step['cores'] = cores
    step['max_memory'] = max_memory
    step['max_disk'] = max_disk
    if min_cores is None:
        min_cores = cores
    step['min_cores'] = min_cores
    if testcase_module is not None:
        step['testcase_module'] = testcase_module
    else:
        if namelist_file is not None or streams_file is not None:
            raise ValueError('You must supply a testcase module for the '
                             'namelist and/or streams file')
    if namelist_file is not None:
        step['namelist'] = namelist_file
    if streams_file is not None:
        step['streams'] = streams_file

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

    # generate the namelist, replacing a few default options
    # start with the same namelist settings as the forward run
    replacements = namelist.parse_replacements(
        'compass.ocean.tests.ice_shelf_2d', 'namelist.forward')

    # we don't want the global stats AM for this run
    replacements['config_AM_globalStats_enable'] = '.false.'

    # we want a shorter run and no freshwater fluxes under the ice shelf from
    # these namelist options
    replacements.update(namelist.parse_replacements(
        'compass.ocean.namelists', 'namelist.ssh_adjust'))

    if 'testcase_module' in step:
        testcase_module = step['testcase_module']
    else:
        testcase_module = None

    # see if there's one for the step itself
    if 'namelist' in step:
        replacements.update(namelist.parse_replacements(
            testcase_module, step['namelist']))

    namelist.generate(config=config, replacements=replacements,
                      step_work_dir=step_dir, core='ocean', mode='forward')

    # generate the streams file
    streams_data = streams.read('compass.ocean.streams', 'streams.ssh_adjust')

    # see if there's one for the testcase itself
    if 'streams' in step:
        streams_data = streams.read(testcase_module, step['streams'],
                                    tree=streams_data)

    streams.generate(config=config, tree=streams_data, step_work_dir=step_dir,
                     core='ocean', mode='forward')

    # make a link to the ocean_model executable
    symlink(os.path.abspath(config.get('executables', 'model')),
            os.path.join(step_dir, 'ocean_model'))

    inputs = []
    outputs = []

    mesh_path = '{}/mesh/mesh'.format(get_mesh_relative_path(step))
    init_path = '{}/init/initial_state'.format(
        get_initial_condition_relative_path(step))

    links = {'{}/initial_state.nc'.format(init_path): 'adjusting_init0.nc',
             '{}/init_mode_forcing_data.nc'.format(init_path):
                 'forcing_data.nc',
             '{}/culled_graph.info'.format(mesh_path): 'graph.info'}
    for target, link in links.items():
        symlink(target, os.path.join(step_dir, link))
        inputs.append(os.path.abspath(os.path.join(step_dir, target)))

    outputs.append(os.path.abspath(os.path.join(step_dir, 'adjusted_init.nc')))

    step['inputs'] = inputs
    step['outputs'] = outputs


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
    cores = step['cores']
    step_dir = step['work_dir']
    iteration_count = config.getint('ssh_adjustment', 'iterations')
    update_namelist_pio(config, cores, step_dir)
    adjust_ssh(variable='landIcePressure', iteration_count=iteration_count,
               config=config, cores=cores, logger=logger)
