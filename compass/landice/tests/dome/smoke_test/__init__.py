from compass.testcase import add_step, run_steps, set_testcase_subdir
from compass.landice.tests.dome import setup_mesh, run_model, visualize


def collect(testcase):
    """
    Update the dictionary of test case properties and add steps

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case, which can be updated
    """
    mesh_type = testcase['mesh_type']
    testcase['description'] = \
        'dome - {} - smoke test'.format(mesh_type.replace('_', ' '))

    subdir = '{}/{}'.format(mesh_type, testcase['name'])
    set_testcase_subdir(testcase, subdir)

    add_step(testcase, setup_mesh, mesh_type=mesh_type)

    add_step(testcase, run_model, cores=4, threads=1, mesh_type=mesh_type)

    add_step(testcase, visualize, mesh_type=mesh_type)

    # we don't want to run the visualize step by default.  The user will do
    # this manually if they want viz
    testcase['steps_to_run'] = ['setup_mesh', 'run_model']


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
