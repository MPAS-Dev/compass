import os

from compass.ocean.tests.global_ocean.metadata import \
    add_mesh_and_init_metadata
from compass.io import symlink, add_input_file, add_output_file
from compass.namelist import add_namelist_file, generate_namelist
from compass.streams import add_streams_file, generate_streams
from compass.model import add_model_as_input, run_model
from compass.ocean.vertical import generate_grid, write_grid
from compass.ocean.plot import plot_vertical_grid, plot_initial_state
from compass.ocean.tests.global_ocean.subdir import get_mesh_relative_path


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
    initial_condition = step['initial_condition']
    if initial_condition not in ['PHC', 'EN4_1900']:
        raise ValueError('Unknown initial_condition {}'.format(
            initial_condition))


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
    with_ice_shelf_cavities = step['with_ice_shelf_cavities']
    initial_condition = step['initial_condition']
    with_bgc = step['with_bgc']
    package = 'compass.ocean.tests.global_ocean.init'

    # generate the namelist, replacing a few default options
    add_namelist_file(step, package, 'namelist.init')
    add_namelist_file(step, package,
                      'namelist.{}'.format(initial_condition.lower()))
    if with_ice_shelf_cavities:
        add_namelist_file(step, package, 'namelist.wisc')
    if with_bgc:
        add_namelist_file(step, package, 'namelist.bgc')

    generate_namelist(step, config, mode='init')

    # generate the streams file
    add_streams_file(step, package, 'streams.init')

    if with_ice_shelf_cavities:
        add_streams_file(step, package, 'streams.wisc')

    generate_streams(step, config, mode='init')

    add_input_file(
        step,  filename='topography.nc',
        target='BedMachineAntarctica_and_GEBCO_2019_0.05_degree.200128.nc',
        database='bathymetry_database')

    add_input_file(
        step,  filename='wind_stress.nc',
        target='windStress.ncep_1958-2000avg.interp3600x2431.151106.nc',
        database='initial_condition_database')

    add_input_file(step,  filename='swData.nc',
                   target='chlorophyllA_monthly_averages_1deg.151201.nc',
                   database='initial_condition_database')

    if initial_condition == 'PHC':
        add_input_file(
            step, filename='temperature.nc',
            target='PotentialTemperature.01.filled.60levels.PHC.151106.nc',
            database='initial_condition_database')
        add_input_file(step, filename='salinity.nc',
                       target='Salinity.01.filled.60levels.PHC.151106.nc',
                       database='initial_condition_database')
    else:
        # EN4_1900
        add_input_file(
            step, filename='temperature.nc',
            target='PotentialTemperature.100levels.Levitus.EN4_1900estimate.'
                   '200813.nc',
            database='initial_condition_database')
        add_input_file(
            step, filename='salinity.nc',
            target='Salinity.100levels.Levitus.EN4_1900estimate.200813.nc',
            database='initial_condition_database')

    if with_bgc:
        add_input_file(
            step, filename='ecosys.nc',
            target='ecosys_jan_IC_360x180x60_corrO2_Dec2014phaeo.nc',
            database='initial_condition_database')
        add_input_file(
            step, filename='ecosys_forcing.nc',
            target='ecoForcingAllSurface.forMPASO.interp360x180.1timeLevel.nc',
            database='initial_condition_database')

    mesh_path = '{}/mesh/mesh'.format(get_mesh_relative_path(step))

    add_input_file(step, filename='mesh.nc',
                   target='{}/culled_mesh.nc'.format(mesh_path))

    add_input_file(
        step, filename='critical_passages.nc',
        target='{}/critical_passages_mask_final.nc'.format(mesh_path))

    add_input_file(step, filename='graph.info',
                   target='{}/culled_graph.info'.format(mesh_path))

    if with_ice_shelf_cavities:
        add_input_file(step, filename='land_ice_mask.nc',
                       target='{}/land_ice_mask.nc'.format(mesh_path))

    add_model_as_input(step, config)

    for file in ['initial_state.nc', 'init_mode_forcing_data.nc']:
        add_output_file(step, filename=file)

    # get the these properties from the config options
    for option in ['cores', 'min_cores', 'max_memory', 'max_disk',
                   'threads']:
        step[option] = config.getint('global_ocean',
                                     'init_{}'.format(option))


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
    interfaces = generate_grid(config=config)

    write_grid(interfaces=interfaces, out_filename='vertical_grid.nc')
    plot_vertical_grid(grid_filename='vertical_grid.nc', config=config,
                       out_filename='vertical_grid.png')

    run_model(step, config, logger)

    add_mesh_and_init_metadata(step['outputs'], config,
                               init_filename='initial_state.nc')

    plot_initial_state(input_file_name='initial_state.nc',
                       output_file_name='initial_state.png')
