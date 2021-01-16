import os
from importlib.resources import contents

from compass.testcase import get_step_default
from compass.io import symlink
from compass import namelist, streams
from compass.model import partition, run_model
from compass.parallel import update_namelist_pio
from compass.ocean.tests.global_ocean.mesh.mesh import get_mesh_package
from compass.ocean.tests.global_ocean.subdir import get_mesh_relative_path, \
    get_initial_condition_relative_path
from compass.ocean.tests.global_ocean.metadata import \
    add_mesh_and_init_metadata


def collect(mesh_name, with_ice_shelf_cavities, with_bgc, time_integrator,
            cores=None, min_cores=None, max_memory=None, max_disk=None,
            threads=None, testcase_module=None, namelist_file=None,
            streams_file=None, inputs=None, outputs=None,
            namelist_replacements=None, stream_replacements=None):
    """
    Get a dictionary of step properties

    Parameters
    ----------
    mesh_name : str
        The name of the mesh

    with_ice_shelf_cavities : bool
        Whether the mesh should include ice-shelf cavities

    with_bgc : bool, optional
        Whether BGC variables are included in the initial condition

    time_integrator : {'split_explicit', 'RK4'}, optional
        The time integrator to use for the run

    cores : int, optional
        The number of cores to run on in forward runs. If this many cores are
        available on the machine or batch job, the task will run on that
        number. If fewer are available (but no fewer than min_cores), the job
        will run on all available cores instead.  If ``cores`` is not supplied,
        it will be read from the ``forward_cores`` config option in the
        ``global_ocean`` section.

    min_cores : int, optional
        The minimum allowed cores.  If that number of cores are not available
        on the machine or in the batch job, the run will fail.  If
        ``min_cores`` is not supplied, but ``cores`` is, ``min_cores = cores``.
        If neither is supplied, ``min_cores`` will be read from the
        ``forward_min_cores`` config option in the ``global_ocean`` section.

    max_memory : int, optional
        The maximum amount of memory (in MB) this step is allowed to use. If
        ``max_memory`` is not supplied, it will be read from the
        ``forward_max_memory`` config option in the ``global_ocean`` section.

    max_disk : int, optional
        The maximum amount of disk space  (in MB) this step is allowed to use.
        If  ``max_disk`` is not supplied, it will be read from the
        ``forward_max_disk`` config option in the ``global_ocean`` section.

    threads : int, optional
        The number of threads to run with during forward runs. If  ``threads``
         is not supplied, it will be read from the ``forward_threads`` config
         option in the ``global_ocean`` section.

    testcase_module : str, optional
        The module for the testcase

    namelist_file : str, optional
        The name of a namelist file in the testcase package directory

    streams_file : str, optional
        The name of a streams file in the testcase package directory

    inputs : list, optional
        List of relative paths to input files for the step.  In addition to
        this list, the graph file, initial condition and initial forcing are
        always included as inputs.  No local symlinks within the step folder
        are created to these inputs.

    outputs : list, optional
        List of relative paths to output files within the step, the default
        is ``['output.nc']``

    namelist_replacements : dict, optional
        A dictionary of namelist options and values that take priority over all
        other namelist options

    stream_replacements : dict, optional
        If present, ``streams_file`` is treated as a template and these
        replacements are used to fill in the template.

    Returns
    -------
    step : dict
        A dictionary of properties of this step
    """
    step = get_step_default(__name__)
    step['mesh_name'] = mesh_name
    if cores is not None:
        step['cores'] = cores
        if min_cores is None:
            step['min_cores'] = cores
    if max_memory is not None:
        step['max_memory'] = max_memory
    if max_disk is not None:
        step['max_disk'] = max_disk
    if min_cores is not None:
        step['min_cores'] = min_cores
    if threads is not None:
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

    step['with_ice_shelf_cavities'] = with_ice_shelf_cavities
    step['with_bgc'] = with_bgc
    step['time_integrator'] = time_integrator

    if outputs is None:
        outputs = ['output.nc']
    step['outputs'] = outputs

    if inputs is not None:
        step['inputs'] = inputs

    if namelist_replacements is not None:
        step['namelist_replacements'] = namelist_replacements

    if stream_replacements is not None:
        if streams_file is None:
            raise ValueError('if streams replacements are provided, a template'
                             ' must be provided in the streams file.')
        step['stream_replacements'] = stream_replacements

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
    mesh_name = step['mesh_name']
    step_dir = step['work_dir']
    with_ice_shelf_cavities = step['with_ice_shelf_cavities']
    with_bgc = step['with_bgc']
    time_integrator = step['time_integrator']

    # generate the namelist, replacing a few default options
    replacements = dict()

    replacements.update(namelist.parse_replacements(
        'compass.ocean.tests.global_ocean', 'namelist.forward'))

    if with_ice_shelf_cavities:
        replacements.update(namelist.parse_replacements(
            'compass.ocean.tests.global_ocean', 'namelist.wisc'))

    if with_bgc:
        replacements.update(namelist.parse_replacements(
            'compass.ocean.tests.global_ocean', 'namelist.bgc'))

    if 'testcase_module' in step:
        testcase_module = step['testcase_module']
    else:
        testcase_module = None

    # add forward namelist options for this mesh
    mesh_package, _ = get_mesh_package(mesh_name)
    mesh_package_contents = list(contents(mesh_package))
    mesh_namelists = ['namelist.forward',
                      'namelist.{}'.format(time_integrator.lower())]
    for mesh_namelist in mesh_namelists:
        if mesh_namelist in mesh_package_contents:
            replacements.update(namelist.parse_replacements(
                mesh_package, mesh_namelist))

    if 'namelist.forward' in mesh_package_contents:
        replacements.update(namelist.parse_replacements(
            mesh_package, 'namelist.forward'))

    # see if there's one for the testcase itself
    if 'namelist' in step:
        replacements.update(namelist.parse_replacements(
            testcase_module, step['namelist']))

    # finally, add or update any replacements passed into collect()
    if 'namelist_replacements' in step:
        replacements.update(step['namelist_replacements'])

    namelist.generate(config=config, replacements=replacements,
                      step_work_dir=step_dir, core='ocean', mode='forward')

    # generate the streams file
    streams_data = streams.read('compass.ocean.tests.global_ocean',
                                'streams.forward')

    if with_bgc:
        streams_data = streams.read('compass.ocean.tests.global_ocean',
                                    'streams.bgc', tree=streams_data)

    # add streams for the mesh
    mesh_streams = ['streams.forward',
                    'streams.{}'.format(time_integrator.lower())]
    for mesh_stream in mesh_streams:
        if mesh_stream in mesh_package_contents:
            streams.read(mesh_package, mesh_stream, tree=streams_data)

    # see if there's one for the testcase itself
    if 'stream_replacements' in step:
        stream_replacements = step['stream_replacements']
    else:
        stream_replacements = None
    if 'streams' in step:
        streams_data = streams.read(testcase_module, step['streams'],
                                    tree=streams_data,
                                    replacements=stream_replacements)

    streams.generate(config=config, tree=streams_data, step_work_dir=step_dir,
                     core='ocean', mode='forward')

    # make a link to the ocean_model executable
    symlink(os.path.abspath(config.get('executables', 'model')),
            os.path.join(step_dir, 'ocean_model'))

    if 'inputs' in step:
        inputs = [os.path.join(step_dir, file) for file in step['inputs']]
    else:
        inputs = []

    mesh_path = '{}/mesh/mesh'.format(get_mesh_relative_path(step))
    init_path = '{}/init'.format(get_initial_condition_relative_path(step))

    if with_ice_shelf_cavities:
        initial_state_target = '{}/ssh_adjustment/adjusted_init.nc'.format(
            init_path)
    else:
        initial_state_target = '{}/initial_state/initial_state.nc'.format(
            init_path)
    links = {initial_state_target: 'init.nc',
             '{}/initial_state/init_mode_forcing_data.nc'.format(init_path):
                 'forcing_data.nc',
             '{}/culled_graph.info'.format(mesh_path): 'graph.info'}

    for target, link in links.items():
        symlink(target, os.path.join(step_dir, link))
        inputs.append(os.path.abspath(os.path.join(step_dir, target)))

    step['inputs'] = inputs

    # convert from relative to absolute paths
    step['outputs'] = [os.path.abspath(os.path.join(step_dir, file)) for file
                       in step['outputs']]


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
    threads = step['threads']
    step_dir = step['work_dir']
    update_namelist_pio(config, cores, step_dir)
    partition(cores, logger)

    run_model(config, core='ocean', core_count=cores, logger=logger,
              threads=threads)

    add_mesh_and_init_metadata(step['outputs'], config,
                               init_filename='init.nc')
