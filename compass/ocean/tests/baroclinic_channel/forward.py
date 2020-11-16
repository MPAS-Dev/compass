import os

from compass.step import get_default
from compass.io import symlink
from compass import namelist, streams
from compass.model import partition, run_model
from compass.ocean.tests.baroclinic_channel.parallel import get_core_count


def collect(resolution, procs, threads, testcase_module=None,
            namelist_file=None, streams_file=None, nu=None):
    """
    Get a dictionary of step properties

    Parameters
    ----------
    resolution : {'1km', '4km', '10km'}
        The name of the resolution to run at

    procs : int
        The number of cores to run on in forward runs

    threads : int
        The number of threads to run with during forward runs

    testcase_module : str, optional
        The module for the testcase

    namelist_file : str, optional
        The name of a namelist file in the testcase package directory

    streams_file : str, optional
        The name of a streams file in the testcase package directory

    nu : float, optional
        The viscosity for this step

    Returns
    -------
    step : dict
        A dictionary of properties of this step
    """
    step = get_default(__name__)
    step['resolution'] = resolution
    step['procs'] = procs
    step['threads'] = threads
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

    if nu is not None:
        step['nu'] = nu

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
    resolution = step['resolution']
    step_dir = step['work_dir']

    # generate the namelist, replacing a few default options
    replacements = dict()

    for namelist_file in ['namelist.forward',
                          'namelist.{}.forward'.format(resolution)]:
        replacements.update(namelist.parse_replacements(
            'compass.ocean.tests.baroclinic_channel', namelist_file))

    if 'testcase_module' in step:
        testcase_module = step['testcase_module']
    else:
        testcase_module = None

    # see if there's one for the testcase itself
    if 'namelist' in step:
        replacements.update(namelist.parse_replacements(
            testcase_module, step['namelist']))

    if 'nu' in step:
        # update the viscosity to the requested value
        replacements.update({'config_mom_del2': '{}'.format(step['nu'])})

    namelist.generate(config=config, replacements=replacements,
                      step_work_dir=step_dir, core='ocean', mode='forward')

    # generate the streams file
    streams_data = streams.read('compass.ocean.tests.baroclinic_channel',
                                'streams.forward')

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

    links = {'../initial_state/ocean.nc': 'init.nc',
             '../initial_state/culled_graph.info': 'graph.info'}
    for target, link in links.items():
        symlink(target, os.path.join(step_dir, link))
        inputs.append(os.path.abspath(os.path.join(step_dir, target)))

    for file in ['output.nc']:
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
    procs = step['procs']
    threads = step['threads']
    step_dir = step['work_dir']
    procs = get_core_count(config, procs, step_dir)
    partition(procs)
    run_model(config, core='ocean', core_count=procs, threads=threads)
