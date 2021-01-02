from compass.testcase import run_steps, get_testcase_default
from compass.ocean.tests.global_ocean.init import mesh, initial_state
from compass.ocean.tests import global_ocean
from compass.validate import compare_variables
from compass.config import add_config


def collect(mesh_name, include_bgc=False):
    """
    Get a dictionary of testcase properties

    Parameters
    ----------
    mesh_name : str
        The name of the mesh

    include_bgc : bool, optional
        Whether to include an initial condition with BGC variables

    Returns
    -------
    testcase : dict
        A dict of properties of this test case, including its steps
    """
    description = 'global ocean {} - init test'.format(mesh_name)
    module = __name__

    name = module.split('.')[-1]
    subdir = '{}/{}'.format(mesh_name, name)
    steps = dict()
    step = mesh.collect(mesh_name, cores=4, min_cores=2,
                        max_memory=1000, max_disk=1000, threads=1)
    steps[step['name']] = step
    step = initial_state.collect(mesh_name=mesh_name, cores=4,
                                 min_cores=2, max_memory=1000, max_disk=1000,
                                 threads=1)
    steps[step['name']] = step

    if include_bgc:
        step = initial_state.collect(mesh_name=mesh_name, cores=4,
                                     min_cores=2, max_memory=1000, max_disk=1000,
                                     threads=1, with_bgc=True)
        step['name'] = 'initial_state_bgc'
        step['subdir'] = step['name']
        steps[step['name']] = step

    testcase = get_testcase_default(module, description, steps, subdir=subdir)
    testcase['mesh_name'] = mesh_name

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
    mesh_lower = mesh_name.lower()
    add_config(config,
               'compass.ocean.tests.global_ocean.mesh.{}'.format(mesh_lower),
               '{}.cfg'.format(mesh_lower), exception=True)


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
    include_bgc = 'initial_state_bgc' in testcase['steps']
    steps = ['mesh', 'initial_state']
    if include_bgc:
        steps.append('initial_state_bgc')

    run_steps(testcase, test_suite, config, steps, logger)
    variables = ['temperature', 'salinity', 'layerThickness']
    compare_variables(variables, config, work_dir,
                      filename1='initial_state/initial_state.nc')

    if include_bgc:
        variables = ['temperature', 'salinity', 'layerThickness', 'PO4', 'NO3',
                     'SiO3', 'NH4', 'Fe', 'O2', 'DIC', 'DIC_ALT_CO2', 'ALK',
                     'DOC', 'DON', 'DOFe', 'DOP', 'DOPr', 'DONr', 'zooC',
                     'spChl', 'spC', 'spFe', 'spCaCO3', 'diatChl', 'diatC',
                     'diatFe', 'diatSi', 'diazChl', 'diazC', 'diazFe',
                     'phaeoChl', 'phaeoC', 'phaeoFe', 'DMS', 'DMSP', 'PROT',
                     'POLY', 'LIP']
        compare_variables(variables, config, work_dir,
                          filename1='initial_state_bgc/initial_state.nc')
