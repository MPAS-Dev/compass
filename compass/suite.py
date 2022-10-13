import argparse
import sys
from importlib import resources

from compass.setup import setup_cases
from compass.clean import clean_cases


def setup_suite(mpas_core, suite_name, config_file=None, machine=None,
                work_dir=None, baseline_dir=None, mpas_model_path=None,
                copy_executable=False):
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
        Location of baselines that can be compared to

    mpas_model_path : str, optional
        The relative or absolute path to the root of a branch where the MPAS
        model has been built

    copy_executable : bool, optional
        Whether to copy the MPAS executable to the work directory
    """
    text = resources.read_text('compass.{}.suites'.format(mpas_core),
                               '{}.txt'.format(suite_name))

    tests, cached = _parse_suite(text)

    setup_cases(tests, config_file=config_file, machine=machine,
                work_dir=work_dir, baseline_dir=baseline_dir,
                mpas_model_path=mpas_model_path, suite_name=suite_name,
                cached=cached, copy_executable=copy_executable)


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

    tests, _ = _parse_suite(text)

    clean_cases(tests=tests, work_dir=work_dir, suite_name=suite_name)


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
    parser.add_argument("--copy_executable", dest="copy_executable",
                        action="store_true",
                        help="If the MPAS executable should be copied to the "
                             "work directory")
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
                    mpas_model_path=args.mpas_model,
                    copy_executable=args.copy_executable)


def _parse_suite(text):
    """ Parse the text of a file defining a test suite """

    tests = list()
    cached = list()
    for test in text.split('\n'):
        test = test.strip()
        if len(test) == 0 or test.startswith('#'):
            # a blank line or comment
            continue

        if test == 'cached':
            cached[-1] = ['_all']
        elif test.startswith('cached:'):
            steps = test[len('cached:'):].strip().split(' ')
            cached[-1].extend(steps)
        else:
            tests.append(test)
            cached.append(list())

    return tests, cached
