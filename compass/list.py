import argparse
import re
import sys
import os
from importlib import resources
from importlib.resources import contents

from compass.mpas_cores import get_mpas_cores


def list_cases(test_expr=None, number=None, verbose=False):
    """
    List the available test cases

    Parameters
    ----------
    test_expr : str, optional
        A regular expression for a test path name to search for

    number : int, optional
        The number of the test to list

    verbose : bool, optional
        Whether to print details of each test or just the subdirectories.
        When applied to suites, verbose will list the tests in the suite.
    """
    mpas_cores = get_mpas_cores()

    if number is None:
        print('Testcases:')

    test_cases = []
    for mpas_core in mpas_cores:
        for test_group in mpas_core.test_groups.values():
            for test_case in test_group.test_cases.values():
                test_cases.append(test_case)

    for test_number, test_case in enumerate(test_cases):
        print_number = False
        print_test = False
        if number is not None:
            if number == test_number:
                print_test = True
        elif test_expr is None or re.match(test_expr, test_case.path):
            print_test = True
            print_number = True

        if print_test:
            number_string = '{:d}: '.format(test_number).rjust(6)
            if print_number:
                prefix = number_string
            else:
                prefix = ''
            if verbose:
                lines = list()
                to_print = {'path': test_case.path,
                            'name': test_case.name,
                            'MPAS core': test_case.mpas_core.name,
                            'test group': test_case.test_group.name,
                            'subdir': test_case.subdir}
                for key in to_print:
                    key_string = '{}: '.format(key).ljust(15)
                    lines.append('{}{}{}'.format(prefix, key_string,
                                                 to_print[key]))
                    if print_number:
                        prefix = '      '
                lines.append('{}steps:'.format(prefix))
                for step in test_case.steps.values():
                    if step.name == step.subdir:
                        lines.append('{} - {}'.format(prefix, step.name))
                    else:
                        lines.append('{} - {}: {}'.format(prefix, step.name,
                                                          step.subdir))
                lines.append('')
                print_string = '\n'.join(lines)
            else:
                print_string = '{}{}'.format(prefix, test_case.path)

            print(print_string)


def list_machines():
    machine_configs = contents('compass.machines')
    print('Machines:')
    for config in machine_configs:
        if config.endswith('.cfg'):
            print('   {}'.format(os.path.splitext(config)[0]))


def list_suites(cores=None, verbose=False):
    if cores is None:
        cores = [mpas_core.name for mpas_core in get_mpas_cores()]
    print('Suites:')
    for core in cores:
        try:
            suites = contents('compass.{}.suites'.format(core))
        except FileNotFoundError:
            continue
        for suite in sorted(suites):
            print_header_if_cached = True
            if suite.endswith('.txt'):
                print('  -c {} -t {}'.format(core, os.path.splitext(suite)[0]))
                if verbose:
                    text = resources.read_text(
                        'compass.{}.suites'.format(core), suite)
                    tests = list()
                    for test in text.split('\n'):
                        test = test.strip()
                        if test == 'cached':
                            print('\t  Cached steps: all')
                        elif test.startswith('cached'):
                            if print_header_if_cached:
                                print('\t  Cached steps:')
                            # don't print the header again if there are
                            # multiple lines of cached steps
                            print_header_if_cached = False
                            print(test.replace('cached: ', '\t    '))
                        elif (len(test) > 0 and test not in tests
                                and not test.startswith('#')):
                            print("\t* {}".format(test))
                            print_header_if_cached = True


def main():
    parser = argparse.ArgumentParser(
        description='List the available test cases or machines',
        prog='compass list')
    parser.add_argument("-t", "--test_expr", dest="test_expr",
                        help="A regular expression for a test path name to "
                             "search for",
                        metavar="TEST")
    parser.add_argument("-n", "--number", dest="number", type=int,
                        help="The number of the test to list")
    parser.add_argument("--machines", dest="machines", action="store_true",
                        help="List supported machines (instead of test cases)")
    parser.add_argument("--suites", dest="suites", action="store_true",
                        help="List test suites (instead of test cases)")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="List details of each test case, not just the "
                             "path.  When applied to suites, verbose lists "
                             "the tests contained in each suite.")
    args = parser.parse_args(sys.argv[2:])
    if args.machines:
        list_machines()
    elif args.suites:
        list_suites(verbose=args.verbose)
    else:
        list_cases(test_expr=args.test_expr, number=args.number,
                   verbose=args.verbose)
