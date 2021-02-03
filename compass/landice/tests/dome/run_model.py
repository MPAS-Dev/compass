from compass.io import add_input_file, add_output_file
from compass.namelist import add_namelist_file, generate_namelist
from compass.streams import add_streams_file, generate_streams
from compass.model import add_model_as_input, run_model


def collect(testcase, step):
    """
    Update the dictionary of step properties

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case, which should not be
        modified here

    step : dict
        A dictionary of properties of this step, which can be updated
    """
    defaults = dict(max_memory=1000, max_disk=1000, threads=1,
                    suffixes=['landice'])
    for key, value in defaults.items():
        step.setdefault(key, value)

    step.setdefault('min_cores', step['cores'])

    # most runs will just have a namelist.landice and a streams.landice but
    # the restart_run step of the restart_test runs the model twice, the second
    # time with namelist.landice.rst and streams.landice.rst
    for suffix in step['suffixes']:
        add_namelist_file(
            step, 'compass.landice.tests.dome', 'namelist.landice',
            out_name='namelist.{}'.format(suffix))

        add_streams_file(
            step, 'compass.landice.tests.dome', 'streams.landice',
            out_name='streams.{}'.format(suffix))

    add_input_file(step, filename='landice_grid.nc',
                   target='../setup_mesh/landice_grid.nc')
    add_input_file(step, filename='graph.info',
                   target='../setup_mesh/graph.info')

    add_output_file(step, filename='output.nc')


def setup(step, config):
    """
    Set up the test case in the work directory, including downloading any
    dependencies

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step from the ``collect()`` function

    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core, configuration and test case
    """
    # again, most runs will just have a namelist.landice and a streams.landice
    # but the restart_run step of the restart_test runs the model twice, the
    # second time with namelist.landice.rst and streams.landice.rst
    for suffix in step['suffixes']:
        generate_namelist(step, config, out_name='namelist.{}'.format(suffix))
        generate_streams(step, config, out_name='streams.{}'.format(suffix))

    add_model_as_input(step, config)


def run(step, test_suite, config, logger):
    """
    Run this step of the test case

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step from the ``collect()``
        function, with modifications from the ``setup()`` function.

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step
    """

    # again, most runs will just have a namelist.landice and a streams.landice
    # but the restart_run step of the restart_test runs the model twice, the
    # second time with namelist.landice.rst and streams.landice.rst
    for suffix in step['suffixes']:
        run_model(step, config, logger, namelist='namelist.{}'.format(suffix),
                  streams='streams.{}'.format(suffix))
