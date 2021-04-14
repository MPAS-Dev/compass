import argparse
import sys
import os

from compass.suite import run_suite
from compass.testcase import run_test_case
from compass.step import run_step


def main():
    parser = argparse.ArgumentParser(
        description='Run a test suite, test case or step',
        prog='compass run')
    parser.add_argument("suite", nargs='?', default=None,
                        help="The name of a test suite to run")
    args = parser.parse_args(sys.argv[2:])
    if args.suite is not None:
        run_suite(args.suite)
    elif os.path.exists('test_case.pickle'):
        run_test_case()
    elif os.path.exists('step.pickle'):
        run_step()
    else:
        raise OSError('A suite name was not given but the current directory '
                      'does not contain a test case or step.')
