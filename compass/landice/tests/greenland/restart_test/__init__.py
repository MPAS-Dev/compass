from compass.testcase import add_step, run_steps
from compass.validate import compare_variables
from compass.namelist import add_namelist_file
from compass.streams import add_streams_file
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
    testcase['description'] = 'GIS {} - restart test'.format(resolution)

    name = 'full_run'
    step = add_step(testcase, run_model, name=name, subdir=name, cores=4,
                    threads=1, resolution=resolution)

    # modify the namelist options and streams file
    add_namelist_file(
        step, 'compass.landice.tests.greenland.restart_test',
        'namelist.full', out_name='namelist.landice')
    add_streams_file(
        step, 'compass.landice.tests.greenland.restart_test',
        'streams.full', out_name='streams.landice')

    name = 'restart_run'
    step = add_step(testcase, run_model, name=name, subdir=name, cores=4,
                    threads=1, resolution=resolution,
                    suffixes=['landice', 'landice.rst'])

    # modify the namelist options and streams file
    add_namelist_file(
        step, 'compass.landice.tests.greenland.restart_test',
        'namelist.restart', out_name='namelist.landice')
    add_streams_file(
        step, 'compass.landice.tests.greenland.restart_test',
        'streams.restart', out_name='streams.landice')

    add_namelist_file(
        step, 'compass.landice.tests.greenland.restart_test',
        'namelist.restart.rst', out_name='namelist.landice.rst')
    # same streams file for both restart stages
    add_streams_file(
        step, 'compass.landice.tests.greenland.restart_test',
        'streams.restart', out_name='streams.landice.rst')


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
    if 'full_run' in steps and 'restart_run' in steps:
        compare_variables(variables, config, work_dir=testcase['work_dir'],
                          filename1='full_run/output.nc',
                          filename2='restart_run/output.nc')
