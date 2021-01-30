import os
from importlib.resources import contents

from compass.io import add_input_file, add_output_file
from compass.namelist import add_namelist_file, generate_namelist
from compass.streams import add_streams_file, generate_streams
from compass.ocean.tests.global_ocean.mesh.mesh import get_mesh_package
from compass.ocean.tests.global_ocean.subdir import get_mesh_relative_path, \
    get_initial_condition_relative_path
from compass.ocean.tests.global_ocean.metadata import \
    add_mesh_and_init_metadata
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
    if 'cores' in step:
        step.setdefault('min_cores', step['cores'])

    mesh_name = step['mesh_name']
    with_ice_shelf_cavities = step['with_ice_shelf_cavities']
    with_bgc = step['with_bgc']
    time_integrator = step['time_integrator']

    add_namelist_file(
        step, 'compass.ocean.tests.global_ocean', 'namelist.forward')
    add_streams_file(
        step, 'compass.ocean.tests.global_ocean', 'streams.forward')

    if with_ice_shelf_cavities:
        add_namelist_file(
            step, 'compass.ocean.tests.global_ocean', 'namelist.wisc')

    if with_bgc:
        add_namelist_file(
            step, 'compass.ocean.tests.global_ocean', 'namelist.bgc')
        add_streams_file(
            step, 'compass.ocean.tests.global_ocean', 'streams.bgc')

    mesh_package, _ = get_mesh_package(mesh_name)
    mesh_package_contents = list(contents(mesh_package))
    mesh_namelists = ['namelist.forward',
                      'namelist.{}'.format(time_integrator.lower())]
    for mesh_namelist in mesh_namelists:
        if mesh_namelist in mesh_package_contents:
            add_namelist_file(step, mesh_package, mesh_namelist)

    mesh_streams = ['streams.forward',
                    'streams.{}'.format(time_integrator.lower())]
    for mesh_stream in mesh_streams:
        if mesh_stream in mesh_package_contents:
            add_streams_file(step, mesh_package, mesh_stream)

    mesh_path = '{}/mesh/mesh'.format(get_mesh_relative_path(step))
    init_path = '{}/init'.format(get_initial_condition_relative_path(step))

    if with_ice_shelf_cavities:
        initial_state_target = '{}/ssh_adjustment/adjusted_init.nc'.format(
            init_path)
    else:
        initial_state_target = '{}/initial_state/initial_state.nc'.format(
            init_path)
    add_input_file(step, filename='init.nc', target=initial_state_target)
    add_input_file(
        step, filename='forcing_data.nc',
        target='{}/initial_state/init_mode_forcing_data.nc'.format(init_path))
    add_input_file(step, filename='graph.info',
                   target='{}/culled_graph.info'.format(mesh_path))

    add_output_file(step, filename='output.nc')


def setup(step, config):
    """
    Set up the test case in the work directory, including downloading any
    dependencies

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step

    config : configparser.ConfigParser
        Configuration options for this test case
    """
    step_dir = step['work_dir']

    generate_namelist(step, config)
    generate_streams(step, config)

    add_model_as_input(step, config)

    for option in ['cores', 'min_cores', 'max_memory', 'max_disk',
                   'threads']:
        if option not in step:
            step[option] = config.getint('global_ocean',
                                         'forward_{}'.format(option))


def run(step, test_suite, config, logger):
    """
    Run this step of the testcase

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this test case

    logger : logging.Logger
        A logger for output from the step
    """
    run_model(step, config, logger)

    add_mesh_and_init_metadata(step['outputs'], config,
                               init_filename='init.nc')
