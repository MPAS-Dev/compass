from compass.testcase import run_steps, get_testcase_default
from compass.ocean.tests.ziso import initial_state, forward
from compass.ocean.tests import ziso
from compass.validate import compare_variables, compare_timers


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
    description = 'Zonally periodic Idealized Southern Ocean (ZISO) {} '\
                  'default test'.format(resolution)
    module = __name__

    res_params = {'20km': {'core_count': 4, 'min_cores': 2,
                           'max_memory': 1000, 'max_disk': 1000}}

    if resolution not in res_params:
        raise ValueError('Unsupported resolution {}. Supported values are: '
                         '{}'.format(resolution, list(res_params)))

    res_params = res_params[resolution]

    name = module.split('.')[-1]
    subdir = '{}/{}'.format(resolution, name)
    steps = dict()
    step = initial_state.collect(resolution)
    steps[step['name']] = step

    # particles are on only for the 20km test case
    if resolution in ['20km']:
        namelist_file = 'namelist.{}.forward'.format(resolution)
    else:
        namelist_file = None
    step = forward.collect(resolution, cores=res_params['core_count'],
                           min_cores=res_params['min_cores'],
                           max_memory=res_params['max_memory'],
                           max_disk=res_params['max_disk'], threads=1,
                           testcase_module=module, namelist_file=namelist_file)
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
    ziso.configure(testcase, config)


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
    steps = ['initial_state', 'forward']
    run_steps(testcase, test_suite, config, steps, logger)
    variables = ['temperature', 'layerThickness']
    compare_variables(variables, config, work_dir,
                      filename1='forward/output/output.0001-01-01_00.00.00.nc')

    variables = ['xParticle', 'yParticle', 'zParticle', 'zLevelParticle',
                 'buoyancyParticle', 'indexToParticleID', 'currentCell',
                 'transfered', 'numTimesReset']
    compare_variables(variables, config, work_dir,
                      filename1='forward/analysis_members/'
                                'lagrPartTrack.0001-01-01_00.00.00.nc')

    timers = ['init_lagrPartTrack', 'compute_lagrPartTrack',
              'write_lagrPartTrack', 'restart_lagrPartTrack',
              'finalize_lagrPartTrack']
    compare_timers(timers, config, work_dir, rundir1='forward')
