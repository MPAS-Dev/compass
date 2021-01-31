from compass.io import add_input_file, add_output_file
from compass.namelist import add_namelist_file, add_namelist_options,\
    generate_namelist
from compass.streams import add_streams_file, generate_streams
from compass.model import add_model_as_input, partition, run_model
from compass.ocean import particles


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
    resolution = step['resolution']
    with_analysis = step['with_analysis']
    with_frazil = step['with_frazil']

    defaults = dict(max_memory=1000, max_disk=1000, threads=1)
    for key, value in defaults.items():
        step.setdefault(key, value)

    step.setdefault('min_cores', step['cores'])

    add_namelist_file(step, 'compass.ocean.tests.ziso', 'namelist.forward')
    add_streams_file(step, 'compass.ocean.tests.ziso', 'streams.forward')

    if with_analysis:
        add_namelist_file(step, 'compass.ocean.tests.ziso',
                          'namelist.analysis')
        add_streams_file(step, 'compass.ocean.tests.ziso', 'streams.analysis')

    if with_frazil:
        add_namelist_options(step,
                             {'config_use_frazil_ice_formation': '.true.'})
        add_streams_file(step, 'compass.ocean.streams', 'streams.frazil')

    add_namelist_file(step, 'compass.ocean.tests.ziso',
                      'namelist.{}.forward'.format(resolution))
    add_streams_file(step, 'compass.ocean.tests.ziso',
                     'streams.{}.forward'.format(resolution))

    add_input_file(step, filename='init.nc',
                   target='../initial_state/ocean.nc')
    add_input_file(step, filename='forcing.nc',
                   target='../initial_state/forcing.nc')
    add_input_file(step, filename='graph.info',
                   target='../initial_state/culled_graph.info')

    add_output_file(step, filename='output/output.0001-01-01_00.00.00.nc')

    if with_analysis:
        add_output_file(
            step,
            filename='analysis_members/lagrPartTrack.0001-01-01_00.00.00.nc')


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
    generate_namelist(step, config)
    generate_streams(step, config)

    add_model_as_input(step, config)


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
    cores = step['cores']

    partition(cores, config, logger)
    particles.write(init_filename='init.nc', particle_filename='particles.nc',
                    graph_filename='graph.info.part.{}'.format(cores),
                    types='buoyancy')
    run_model(step, config, logger, partition_graph=False)
