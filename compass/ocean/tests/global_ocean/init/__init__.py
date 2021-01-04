from compass.testcase import run_steps, get_testcase_default
from compass.ocean.tests.global_ocean.mesh.mesh import get_mesh_package
from compass.ocean.tests.global_ocean.init import initial_state, ssh_adjustment
from compass.ocean.tests.global_ocean.subdir import get_init_sudbdir
from compass.ocean.tests import global_ocean
from compass.validate import compare_variables
from compass.config import add_config


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
                                 initial_condition, with_bgc, cores=4,
                                 min_cores=2, max_memory=1000, max_disk=1000,
                                 threads=1)
    steps[step['name']] = step

    if with_ice_shelf_cavities:
        step = ssh_adjustment.collect(mesh_name, cores=4)
        steps[step['name']] = step

    testcase = get_testcase_default(module, description, steps, subdir=subdir)
    testcase['mesh_name'] = mesh_name
    testcase['with_ice_shelf_cavities'] = with_ice_shelf_cavities
    testcase['initial_condition'] = initial_condition
    testcase['with_bgc'] = with_bgc

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
    mesh_name = testcase['mesh_name']
    mesh_package, prefix = get_mesh_package(mesh_name)
    add_config(config, mesh_package, '{}.cfg'.format(prefix), exception=True)


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
    with_ice_shelf_cavities = testcase['with_ice_shelf_cavities']
    with_bgc = testcase['with_bgc']
    steps = ['initial_state']
    if with_ice_shelf_cavities:
        steps.append('ssh_adjustment')

    run_steps(testcase, test_suite, config, steps, logger)
    variables = ['temperature', 'salinity', 'layerThickness']
    compare_variables(variables, config, work_dir,
                      filename1='initial_state/initial_state.nc')

    if with_bgc:
        variables = ['temperature', 'salinity', 'layerThickness', 'PO4', 'NO3',
                     'SiO3', 'NH4', 'Fe', 'O2', 'DIC', 'DIC_ALT_CO2', 'ALK',
                     'DOC', 'DON', 'DOFe', 'DOP', 'DOPr', 'DONr', 'zooC',
                     'spChl', 'spC', 'spFe', 'spCaCO3', 'diatChl', 'diatC',
                     'diatFe', 'diatSi', 'diazChl', 'diazC', 'diazFe',
                     'phaeoChl', 'phaeoC', 'phaeoFe', 'DMS', 'DMSP', 'PROT',
                     'POLY', 'LIP']
        compare_variables(variables, config, work_dir,
                          filename1='initial_state/initial_state.nc')

    if with_ice_shelf_cavities:
        variables = ['ssh', 'landIcePressure']
        compare_variables(variables, config, work_dir,
                          filename1='ssh_adjustment/adjusted_init.nc')
