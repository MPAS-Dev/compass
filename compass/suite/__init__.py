import argparse
import sys
import os
from importlib import resources
import pickle
import configparser
import stat
from jinja2 import Template
import time
import numpy

from mpas_tools.logging import LoggingContext

from compass.setup import setup_cases
from compass.io import symlink
from compass.clean import clean_cases


def setup_suite(core, suite_name, config_file=None, machine=None,
                work_dir=None, baseline_dir=None):
    """
    Set up a test suite

    Parameters
    ----------
    core : str
        The dynamical core ('ocean', 'landice', etc.) of the test suite

    suite_name : str
        The name of the test suite.  A file ``<suite_name>.txt`` must exist
        within the core's ``suites`` package that lists the paths of the tests
        in the suite

    config_file : str, optional
        Configuration file with custom options for setting up and running
        testcases

    machine : str, optional
        The name of one of the machines with defined config options, which can
        be listed with ``compass list --machines``

    work_dir : str, optional
        A directory that will serve as the base for creating testcase
        directories

    baseline_dir : str, optional
        Location of baseslines that can be compared to
    """

    if config_file is None and machine is None:
        raise ValueError('At least one of config_file and machine is needed.')

    text = resources.read_text('compass.{}.suites'.format(core),
                               '{}.txt'.format(suite_name))
    tests = list()
    for test in text.split('\n'):
        test = test.strip()
        if len(test) > 0 and test not in tests:
            tests.append(test)

    if work_dir is None:
        work_dir = os.getcwd()
    work_dir = os.path.abspath(work_dir)

    testcases = setup_cases(tests, config_file=config_file, machine=machine,
                            work_dir=work_dir, baseline_dir=baseline_dir)

    # if compass/__init__.py exists, we're using a local version of the compass
    # package and we'll want to link to that in the tests and steps
    compass_path = os.path.join(os.getcwd(), 'compass')
    if os.path.exists(os.path.join(compass_path, '__init__.py')):
        symlink(compass_path, os.path.join(work_dir, 'compass'))

    test_suite = {'name': suite_name,
                  'testcases': testcases,
                  'work_dir': work_dir}

    # pickle the test or step dictionary for use at runtime
    pickle_file = os.path.join(test_suite['work_dir'],
                               '{}.pickle'.format(suite_name))
    with open(pickle_file, 'wb') as handle:
        pickle.dump(test_suite, handle, protocol=pickle.HIGHEST_PROTOCOL)

    template = Template(resources.read_text('compass.suite', 'suite.template'))
    script = template.render(suite_name=suite_name)

    run_filename = os.path.join(work_dir, '{}.py'.format(suite_name))
    with open(run_filename, 'w') as handle:
        handle.write(script)

    # make sure it has execute permission
    st = os.stat(run_filename)
    os.chmod(run_filename, st.st_mode | stat.S_IEXEC)

    max_cores, max_of_min_cores = _get_required_cores(testcases)

    print('target cores: {}'.format(max_cores))
    print('minimum cores: {}'.format(max_of_min_cores))


def clean_suite(core, suite_name, work_dir=None):
    """
    Clean up a test suite by removing its testcases and run script

    Parameters
    ----------
    core : str
        The dynamical core ('ocean', 'landice', etc.) of the test suite

    suite_name : str
        The name of the test suite.  A file ``<suite_name>.txt`` must exist
        within the core's ``suites`` package that lists the paths of the tests
        in the suite

    work_dir : str, optional
        A directory that will serve as the base for creating testcase
        directories
    """

    text = resources.read_text('compass.{}.suites'.format(core),
                               '{}.txt'.format(suite_name))
    tests = [test.strip() for test in text.split('\n') if
             len(test.strip()) > 0]

    if work_dir is None:
        work_dir = os.getcwd()
    work_dir = os.path.abspath(work_dir)

    clean_cases(tests=tests, work_dir=work_dir)

    # delete the pickle file and run script
    pickle_file = os.path.join(work_dir, '{}.pickle'.format(suite_name))
    run_filename = os.path.join(work_dir, '{}.py'.format(suite_name))

    for filename in [pickle_file, run_filename]:
        try:
            os.remove(filename)
        except OSError:
            pass


def run_suite(suite_name):
    """
    Run the given test suite

    Parameters
    ----------
    suite_name : str
        The name of the test suite
    """
    with open('{}.pickle'.format(suite_name), 'rb') as handle:
        test_suite = pickle.load(handle)

    # start logging to stdout/stderr
    with LoggingContext(suite_name) as logger:

        os.environ['PYTHONUNBUFFERED'] = '1'

        try:
            os.makedirs('case_outputs')
        except OSError:
            pass

        success = True
        cwd = os.getcwd()
        suite_start = time.time()
        test_times = dict()
        for test_name in test_suite['testcases']:
            testcase = test_suite['testcases'][test_name]

            logger.info(' * Running {}'.format(test_name))
            logger.info('           {}'.format(testcase['description']))

            test_name = testcase['path'].replace('/', '_')
            log_filename = '{}/case_outputs/{}.log'.format(cwd, test_name)
            with LoggingContext(test_name, log_filename=log_filename) as \
                    test_logger:
                testcase['log_filename'] = log_filename
                testcase['new_step_log_file'] = False

                os.chdir(testcase['work_dir'])

                config = configparser.ConfigParser(
                    interpolation=configparser.ExtendedInterpolation())
                config.read(testcase['config'])

                run = getattr(sys.modules[testcase['module']], testcase['run'])
                test_start = time.time()
                try:
                    run(testcase, test_suite, config, test_logger)
                    logger.info('    PASS')
                except BaseException:
                    test_logger.exception('Exception raised')
                    logger.error('   FAIL    For more information, see:')
                    logger.error('           case_outputs/{}.log'.format(
                        test_name))
                    success = False
                test_times[test_name] = time.time() - test_start

            logger.info('')
        suite_time = time.time() - suite_start

        os.chdir(cwd)

        logger.info('Test Runtimes:')
        for test_name, test_time in test_times.items():
            mins = int(numpy.floor(test_time / 60.0))
            secs = int(numpy.ceil(test_time - mins * 60))
            logger.info('{:02d}:{:02d} {}'.format(mins, secs, test_name))
        mins = int(numpy.floor(suite_time / 60.0))
        secs = int(numpy.ceil(suite_time - mins * 60))
        logger.info('Total runtime {:02d}:{:02d}'.format(mins, secs))

        if success:
            logger.info('PASS: All passed successfully!')
        else:
            logger.error('FAIL: One or more tests failed, see above.')
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Set up a regression test suite')
    parser.add_argument("-c", "--core", dest="core",
                        help="The core for the test suite",
                        metavar="CORE", required=True)
    parser.add_argument("-t", "--test_suite", dest="test_suite",
                        help="Path to file containing a test suite to setup",
                        metavar="FILE", required=True)
    parser.add_argument("-f", "--config_file", dest="config_file",
                        help="Configuration file for test case setup",
                        metavar="FILE")
    parser.add_argument("-s", "--setup", dest="setup",
                        help="Option to determine if regression suite should "
                             "be setup or not.", action="store_true")
    parser.add_argument("--clean", dest="clean",
                        help="Option to determine if regression suite should "
                             "be cleaned or not.", action="store_true")
    parser.add_argument("-v", "--verbose", dest="verbose",
                        help="Use verbose output from setup_testcase.py",
                        action="store_true")
    parser.add_argument("-m", "--machine", dest="machine",
                        help="The name of the machine for loading machine-"
                             "related config options", metavar="MACH")
    parser.add_argument("-b", "--baseline_dir", dest="baseline_dir",
                        help="Location of baseslines that can be compared to",
                        metavar="PATH")
    parser.add_argument("-w", "--work_dir", dest="work_dir",
                        help="If set, script will setup the test suite in "
                        "work_dir rather in this script's location.",
                        metavar="PATH")
    args = parser.parse_args(sys.argv[2:])

    if not args.clean and not args.setup:
        raise ValueError('At least one of -s/--setup or --clean must be '
                         'specified')

    if args.clean:
        clean_suite(core=args.core, suite_name=args.test_suite,
                    work_dir=args.work_dir)

    if args.setup:
        setup_suite(core=args.core, suite_name=args.test_suite,
                    config_file=args.config_file, machine=args.machine,
                    work_dir=args.work_dir, baseline_dir=args.baseline_dir)


def _get_required_cores(testcases):
    """ Get the maximum number of target cores and the max of min cores """

    max_cores = 0
    max_of_min_cores = 0
    for test_name, testcase in testcases.items():
        for step_name, step in testcase['steps'].items():
            max_cores = max(max_cores, step['cores'])
            max_of_min_cores = max(max_of_min_cores, step['min_cores'])

    return max_cores, max_of_min_cores
