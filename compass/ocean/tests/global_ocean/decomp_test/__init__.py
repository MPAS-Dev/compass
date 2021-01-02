from compass.testcase import run_steps, get_testcase_default
from compass.ocean.tests.global_ocean import forward
from compass.ocean.tests import global_ocean
from compass.validate import compare_variables


def collect(mesh_name, time_integrator):
    """
    Get a dictionary of testcase properties

    Parameters
    ----------
    mesh_name : str
        The name of the mesh

    time_integrator : {'split_explicit', 'RK4'}
        The time integrator to use for the run

    Returns
    -------
    testcase : dict
        A dict of properties of this test case, including its steps
    """
    description = 'global ocean {} - {} decomposition test'.format(
        mesh_name, time_integrator)
    module = __name__

    name = module.split('.')[-1]
    subdir = '{}/{}/{}'.format(mesh_name, name, time_integrator)

    steps = dict()

    for procs in [4, 8]:
        step = forward.collect(mesh_name=mesh_name, cores=procs, threads=1,
                               time_integrator=time_integrator)
        step['name'] = '{}proc'.format(procs)
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
    steps = ['4proc', '8proc']
    run_steps(testcase, test_suite, config, steps, logger)
    variables = ['temperature', 'salinity', 'layerThickness', 'normalVelocity']
    compare_variables(variables, config, work_dir=testcase['work_dir'],
                      filename1='4proc/output.nc',
                      filename2='8proc/output.nc')
