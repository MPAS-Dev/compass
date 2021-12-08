import argparse
import sys
import os
import shutil

from compass.mpas_cores import get_mpas_cores
from compass import provenance


def clean_cases(tests=None, numbers=None, work_dir=None, suite_name='custom'):
    """
    Set up one or more test cases

    Parameters
    ----------
    tests : list of str, optional
        Relative paths for a test cases to clean up

    numbers : list of int, optional
        Case numbers to clean up, as listed from ``compass list``

    work_dir : str, optional
        The base work directory where test cases have been set up

    suite_name : str, optional
        The name of the test suite if tests to clean up belong to a test
        suite or ``'custom'`` if not
    """

    if tests is None and numbers is None:
        raise ValueError('At least one of tests or numbers is needed.')

    if work_dir is None:
        work_dir = os.getcwd()
    work_dir = os.path.abspath(work_dir)

    mpas_cores = get_mpas_cores()
    all_test_cases = dict()
    for mpas_core in mpas_cores:
        for test_group in mpas_core.test_groups.values():
            for test_case in test_group.test_cases.values():
                all_test_cases[test_case.path] = test_case

    test_cases = dict()
    if numbers is not None:
        keys = list(all_test_cases)
        for number in numbers:
            if number >= len(keys):
                raise ValueError('test number {} is out of range.  There are '
                                 'only {} tests.'.format(number, len(keys)))
            path = keys[number]
            test_cases[path] = all_test_cases[path]

    if tests is not None:
        for path in tests:
            if path not in all_test_cases:
                raise ValueError('Test case with path {} is not in '
                                 'the list of test cases'.format(path))
            test_cases[path] = all_test_cases[path]

    provenance.write(work_dir, test_cases)

    print('Cleaning test cases:')
    for path in test_cases.keys():
        print('  {}'.format(path))

        test_case_dir = os.path.join(work_dir, path)
        try:
            shutil.rmtree(test_case_dir)
        except OSError:
            pass

    # delete the pickle file for the test suite (if any)
    pickle_file = os.path.join(work_dir, '{}.pickle'.format(suite_name))

    try:
        os.remove(pickle_file)
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description='Clean up one or more test cases', prog='compass clean')

    parser.add_argument("-t", "--test", dest="test",
                        help="Relative path for a test case to set up",
                        metavar="PATH")
    parser.add_argument("-n", "--case_number", nargs='+', dest="case_num",
                        type=int,
                        help="Case number(s) to setup, as listed from "
                             "'compass list'. Can be a space-separated"
                             "list of case numbers.", metavar="NUM")
    parser.add_argument("-w", "--work_dir", dest="work_dir",
                        help="If set, case directories are created in "
                             "work_dir rather than the current directory.",
                        metavar="PATH")
    args = parser.parse_args(sys.argv[2:])
    if args.test is None:
        tests = None
    else:
        tests = [args.test]
    clean_cases(tests=tests, numbers=args.case_num, work_dir=args.work_dir)
