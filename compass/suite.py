import argparse
import sys
import os
from importlib import resources
import pickle

from compass.setup import setup_cases
from compass.io import symlink
from compass.clean import clean_cases


def setup_suite(mpas_core, suite_name, config_file=None, machine=None,
                work_dir=None, baseline_dir=None, mpas_model_path=None):
    """
    Set up a test suite

    Parameters
    ----------
    mpas_core : str
        The MPAS core ('ocean', 'landice', etc.) of the test suite

    suite_name : str
        The name of the test suite.  A file ``<suite_name>.txt`` must exist
        within the core's ``suites`` package that lists the paths of the tests
        in the suite

    config_file : str, optional
        Configuration file with custom options for setting up and running
        test cases

    machine : str, optional
        The name of one of the machines with defined config options, which can
        be listed with ``compass list --machines``

    work_dir : str, optional
        A directory that will serve as the base for creating test case
        directories

    baseline_dir : str, optional
        Location of baseslines that can be compared to

    mpas_model_path : str, optional
        The relative or absolute path to the root of a branch where the MPAS
        model has been built
    """

    if config_file is None and machine is None:
        raise ValueError('At least one of config_file and machine is needed.')

    text = resources.read_text('compass.{}.suites'.format(mpas_core),
                               '{}.txt'.format(suite_name))
    tests = list()
    for test in text.split('\n'):
        test = test.strip()
        if len(test) > 0 and test not in tests:
            tests.append(test)

    if work_dir is None:
        work_dir = os.getcwd()
    work_dir = os.path.abspath(work_dir)

    test_cases = setup_cases(tests, config_file=config_file, machine=machine,
                             work_dir=work_dir, baseline_dir=baseline_dir,
                             mpas_model_path=mpas_model_path)

    # if compass/__init__.py exists, we're using a local version of the compass
    # package and we'll want to link to that in the tests and steps
    compass_path = os.path.join(os.getcwd(), 'compass')
    if os.path.exists(os.path.join(compass_path, '__init__.py')):
        symlink(compass_path, os.path.join(work_dir, 'compass'))

    test_suite = {'name': suite_name,
                  'test_cases': test_cases,
                  'work_dir': work_dir}

    # pickle the test or step dictionary for use at runtime
    pickle_file = os.path.join(test_suite['work_dir'],
                               '{}.pickle'.format(suite_name))
    with open(pickle_file, 'wb') as handle:
        pickle.dump(test_suite, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if 'LOAD_COMPASS_ENV' in os.environ:
        script_filename = os.environ['LOAD_COMPASS_ENV']
        # make a symlink to the script for loading the compass conda env.
        symlink(script_filename, os.path.join(work_dir, 'load_compass_env.sh'))

    max_cores, max_of_min_cores = _get_required_cores(test_cases)

    print('target cores: {}'.format(max_cores))
    print('minimum cores: {}'.format(max_of_min_cores))


def clean_suite(mpas_core, suite_name, work_dir=None):
    """
    Clean up a test suite by removing its test cases and run script

    Parameters
    ----------
    mpas_core : str
        The MPAS core ('ocean', 'landice', etc.) of the test suite

    suite_name : str
        The name of the test suite.  A file ``<suite_name>.txt`` must exist
        within the core's ``suites`` package that lists the paths of the tests
        in the suite

    work_dir : str, optional
        A directory that will serve as the base for creating test case
        directories
    """

    text = resources.read_text('compass.{}.suites'.format(mpas_core),
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


def main():
    parser = argparse.ArgumentParser(
        description='Set up a regression test suite', prog='compass suite')
    parser.add_argument("-c", "--core", dest="core",
                        help="The MPAS core for the test suite",
                        metavar="CORE", required=True)
    parser.add_argument("-t", "--test_suite", dest="test_suite",
                        help="Path to file containing a test suite to setup",
                        metavar="SUITE", required=True)
    parser.add_argument("-f", "--config_file", dest="config_file",
                        help="Configuration file for test case setup",
                        metavar="FILE")
    parser.add_argument("-s", "--setup", dest="setup",
                        help="Option to determine if regression suite should "
                             "be setup or not.", action="store_true")
    parser.add_argument("--clean", dest="clean",
                        help="Option to determine if regression suite should "
                             "be cleaned or not.", action="store_true")
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
    parser.add_argument("-p", "--mpas_model", dest="mpas_model",
                        help="The path to the build of the MPAS model for the "
                             "core.",
                        metavar="PATH")
    args = parser.parse_args(sys.argv[2:])

    if not args.clean and not args.setup:
        raise ValueError('At least one of -s/--setup or --clean must be '
                         'specified')

    if args.clean:
        clean_suite(mpas_core=args.core, suite_name=args.test_suite,
                    work_dir=args.work_dir)

    if args.setup:
        setup_suite(mpas_core=args.core, suite_name=args.test_suite,
                    config_file=args.config_file, machine=args.machine,
                    work_dir=args.work_dir, baseline_dir=args.baseline_dir,
                    mpas_model_path=args.mpas_model)


def _get_required_cores(test_cases):
    """ Get the maximum number of target cores and the max of min cores """

    max_cores = 0
    max_of_min_cores = 0
    for test_case in test_cases.values():
        for step in test_case.steps.values():
            max_cores = max(max_cores, step.cores)
            max_of_min_cores = max(max_of_min_cores, step.min_cores)

    return max_cores, max_of_min_cores
