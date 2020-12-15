import argparse
import sys
import os
import shutil

import compass.testcases
from compass import provenance


def clean_cases(tests=None, numbers=None, work_dir=None):
    """
    Set up one or more test cases

    Parameters
    ----------
    tests : list of str, optional
        Relative paths for a test cases to set up

    numbers : list of int, optional
        Case numbers to setup, as listed from ``compass list``

    work_dir : str, optional
        A directory that will serve as the base for creating case directories
    """

    if tests is None and numbers is None:
        raise ValueError('At least one of tests or numbers is needed.')

    if work_dir is None:
        work_dir = os.getcwd()

    all_testcases = compass.testcases.collect()
    testcases = dict()
    if numbers is not None:
        keys = list(all_testcases)
        for number in numbers:
            if number >= len(keys):
                raise ValueError('test number {} is out of range.  There are '
                                 'only {} tests.'.format(number, len(keys)))
            path = keys[number]
            testcases[path] = all_testcases[path]

    if tests is not None:
        for path in tests:
            if path not in all_testcases:
                raise ValueError('Testcase with path {} is not in '
                                 'testcases'.format(path))
            testcases[path] = all_testcases[path]

    provenance.write(work_dir, testcases)

    print('Cleaning testcases:')
    for path in testcases.keys():
        print('  {}'.format(path))

        testcase_dir = os.path.join(work_dir, path)
        try:
            shutil.rmtree(testcase_dir)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description='Clean up one or more test cases')

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
