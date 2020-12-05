from compass.testcase import run_steps, get_testcase_default
from compass.ocean.tests.baroclinic_channel import initial_state, forward
from compass.ocean.tests import baroclinic_channel
from compass.validate import compare_variables


def collect(resolution):
    """
    Get a dictionary of testcase properties

    Parameters
    ----------
    resolution : {'10km'}
        The resolution of the mesh

    Returns
    -------
    testcase : dict
        A dict of properties of this test case, including its steps
    """
    description = 'baroclinic channel {} default'.format(resolution)
    module = __name__

    name = module.split('.')[-1]
    subdir = '{}/{}'.format(resolution, name)
    steps = dict()
    step = initial_state.collect(resolution)
    steps[step['name']] = step

    for procs in [4, 8]:
        step = forward.collect(resolution, procs=procs, threads=1)
        step['name'] = '{}proc'.format(procs)
        step['subdir'] = step['name']
        steps[step['name']] = step

    testcase = get_testcase_default(module, description, steps, subdir=subdir)
    testcase['resolution'] = resolution

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
    baroclinic_channel.configure(testcase, config)


def run(testcase, config):
    """
    Run each step of the testcase

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this testcase from the ``collect()``
        function

    config : configparser.ConfigParser
        Configuration options for this testcase, a combination of the defaults
        for the machine, core and configuration
    """
    steps = ['initial_state', '4proc', '8proc']
    run_steps(testcase, config, steps)
    variables = ['temperature', 'salinity', 'layerThickness', 'normalVelocity']
    compare_variables(variables, config, work_dir=testcase['work_dir'],
                      filename1='4proc/output.nc',
                      filename2='8proc/output.nc')
