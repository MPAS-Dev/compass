from compass.testcase import run_steps, get_testcase_default
from compass.ocean.tests.global_ocean.init import initial_state, ssh_adjustment
from compass.ocean.tests.global_ocean.subdir import get_init_sudbdir
from compass.ocean.tests import global_ocean
from compass.validate import compare_variables


def collect(mesh_name, with_ice_shelf_cavities, initial_condition, with_bgc):
    """
    Get a dictionary of testcase properties

    Parameters
    ----------
    mesh_name : str
        The name of the mesh

    with_ice_shelf_cavities : bool
        Whether the mesh should include ice-shelf cavities

    initial_condition : {'PHC', 'EN4_1900'}
        The initial condition to build

    with_bgc : bool
        Whether to include BGC fields in the initial condition

    Returns
    -------
    testcase : dict
        A dict of properties of this test case, including its steps
    """

    module = __name__
    if with_bgc:
        desc_initial_condition = '{} with BGC'.format(initial_condition)
    else:
        desc_initial_condition = initial_condition
    description = 'global ocean {} - {} initial condition'.format(
        mesh_name, desc_initial_condition)

    init_subdir = get_init_sudbdir(mesh_name, initial_condition, with_bgc)

    name = module.split('.')[-1]
    subdir = '{}/{}'.format(init_subdir, name)
    steps = dict()
    step = initial_state.collect(mesh_name, with_ice_shelf_cavities,
                                 initial_condition, with_bgc)
    steps[step['name']] = step

    steps_to_run = ['initial_state']
    if with_ice_shelf_cavities:
        step = ssh_adjustment.collect(mesh_name, cores=4)
        steps[step['name']] = step
        steps_to_run.append('ssh_adjustment')

    testcase = get_testcase_default(module, description, steps, subdir=subdir)
    testcase['mesh_name'] = mesh_name
    testcase['with_ice_shelf_cavities'] = with_ice_shelf_cavities
    testcase['initial_condition'] = initial_condition
    testcase['with_bgc'] = with_bgc
    testcase['steps_to_run'] = steps_to_run

    return testcase


def configure(testcase, config):
    """
    Modify the configuration options for this testcase.

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this testcase from the ``collect()``
        function

    config : configparser.ConfigParser
        Configuration options for this testcase, a combination of the defaults
        for the machine, core and configuration
    """
    global_ocean.configure(testcase, config)


def run(testcase, test_suite, config, logger):
    """
    Run each step of the testcase

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this testcase from the ``collect()``
        function

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this testcase, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the testcase
    """
    work_dir = testcase['work_dir']
    with_bgc = testcase['with_bgc']
    steps = testcase['steps_to_run']
    if 'initial_state' in steps:
        step = testcase['steps']['initial_state']
        # get the these properties from the config options
        for option in ['cores', 'min_cores', 'max_memory', 'max_disk',
                       'threads']:
            step[option] = config.getint('global_ocean',
                                         'init_{}'.format(option))

    if 'ssh_adjustment' in steps:
        step = testcase['steps']['ssh_adjustment']
        # get the these properties from the config options
        for option in ['cores', 'min_cores', 'max_memory', 'max_disk',
                       'threads']:
            step[option] = config.getint('global_ocean',
                                         'forward_{}'.format(option))

    run_steps(testcase, test_suite, config, logger)

    if 'initial_state' in steps:
        variables = ['temperature', 'salinity', 'layerThickness']
        compare_variables(variables, config, work_dir,
                          filename1='initial_state/initial_state.nc')

        if with_bgc:
            variables = ['temperature', 'salinity', 'layerThickness', 'PO4',
                         'NO3', 'SiO3', 'NH4', 'Fe', 'O2', 'DIC',
                         'DIC_ALT_CO2', 'ALK', 'DOC', 'DON', 'DOFe', 'DOP',
                         'DOPr', 'DONr', 'zooC', 'spChl', 'spC', 'spFe',
                         'spCaCO3', 'diatChl', 'diatC', 'diatFe', 'diatSi',
                         'diazChl', 'diazC', 'diazFe', 'phaeoChl', 'phaeoC',
                         'phaeoFe', 'DMS', 'DMSP', 'PROT', 'POLY', 'LIP']
            compare_variables(variables, config, work_dir,
                              filename1='initial_state/initial_state.nc')

    if 'ssh_adjustment' in steps:
        variables = ['ssh', 'landIcePressure']
        compare_variables(variables, config, work_dir,
                          filename1='ssh_adjustment/adjusted_init.nc')


def add_descriptions_to_config(testcase, config):
    """
    Modify the configuration options for this testcase to include a
    description of the initial condition, bathymetry, and whether ice-shelf
    cavities and biogeochemistry were included

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this testcase from the ``collect()``
        function

    config : configparser.ConfigParser
        Configuration options for this testcase, a combination of the defaults
        for the machine, core and configuration
    """
    # add a description of the initial condition
    if 'initial_condition' in testcase:
        initial_condition = testcase['initial_condition']
        descriptions = {'PHC': 'Polar science center Hydrographic Climatology',
                        'EN4_1900':
                            "Met Office Hadley Centre's EN4 dataset from 1900"}
        config.set('global_ocean', 'init_description',
                   descriptions[initial_condition])

    # a description of the bathymetry
    config.set('global_ocean', 'bathy_description',
               'Bathymetry is from GEBCO 2019, combined with BedMachine '
               'Antarctica around Antarctica.')

    if 'with_bgc' in testcase and testcase['with_bgc']:
        # todo: this needs to be filled in!
        config.set('global_ocean', 'bgc_description',
                   '<<<Missing>>>')

    if 'with_ice_shelf_cavities' in testcase and \
            testcase['with_ice_shelf_cavities']:
        config.set('global_ocean', 'wisc_description',
                   'Includes cavities under the ice shelves around Antarctica')
