from compass.testcase import add_step, run_steps
from compass.validate import compare_variables
from compass.landice.tests.greenland import run_model


def collect(testcase):
    """
    Update the dictionary of test case properties and add steps

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case, which can be updated
    """
    resolution = testcase['resolution']
    testcase['description'] = 'GIS {} decomposition test'.format(resolution)

    for procs in [1, 8]:
        name = '{}proc_run'.format(procs)
        add_step(testcase, run_model, name=name, subdir=name, cores=procs,
                 threads=1, resolution=resolution)

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
    variables = ['thickness', 'normalVelocity']
    steps = testcase['steps_to_run']
    if '1proc_run' in steps and '8proc_run' in steps:
        compare_variables(variables, config, work_dir=testcase['work_dir'],
                          filename1='1proc_run/output.nc',
                          filename2='8proc_run/output.nc')
