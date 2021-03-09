from compass.testcase import add_step, run_steps
from compass.namelist import add_namelist_options
from compass.streams import add_streams_file
from compass.validate import compare_variables
from compass.landice.tests.eismint2 import setup_mesh, run_experiment


def collect(testcase):
    """
    Update the dictionary of test case properties and add steps

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case, which can be updated
    """
    testcase['description'] = 'EISMINT2 decomposition test'

    add_step(testcase, setup_mesh)

    experiment = 'f'
    for procs in [1, 4]:
        name = '{}proc_run'.format(procs)
        step = add_step(testcase, run_experiment, name=name, subdir=name,
                        cores=procs, threads=1, experiment=experiment)

        options = {'config_run_duration': "'3000-00-00_00:00:00'"}
        add_namelist_options(step, options)

        add_streams_file(step,
                         'compass.landice.tests.eismint2.decomposition_test',
                         'streams.landice')


# no configure function is needed


def run(testcase, test_suite, config, logger):
    """
    Run each step of the test case

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case from the ``collect()``
        function

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the test case
    """
    run_steps(testcase, test_suite, config, logger)
    variables = ['thickness', 'temperature', 'basalTemperature',
                 'heatDissipation']
    steps = testcase['steps_to_run']
    if '1proc_run' in steps and '4proc_run' in steps:
        compare_variables(variables, config, work_dir=testcase['work_dir'],
                          filename1='1proc_run/output.nc',
                          filename2='4proc_run/output.nc')
