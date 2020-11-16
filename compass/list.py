import argparse
import re
import sys
import os
from importlib.resources import contents

from compass.testcases import collect
from compass import machines


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
        Whether to print details of each test or just the subdirectories
    """
    testcases = collect()

    if number is None:
        print('Testcases:')

    for test_number, (module, test) in enumerate(testcases.items()):
        print_number = False
        print_test = False
        if number is not None:
            if number == test_number:
                print_test = True
        elif test_expr is None or re.match(test_expr, module) or \
                re.match(test_expr, test['path']):
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
                for key in ['path', 'description', 'name', 'core',
                            'configuration', 'subdir']:
                    key_string = '{}: '.format(key).ljust(15)
                    lines.append('{}{}{}'.format(prefix, key_string, test[key]))
                    if print_number:
                        prefix = '      '
                lines.append('{}steps:'.format(prefix))
                for step in test['steps']:
                    lines.append('{} - {}'.format(prefix, step))
                lines.append('')
                print_string = '\n'.join(lines)
            else:
                print_string = '{}{}'.format(prefix, test['path'])

            print(print_string)


def list_machines():
    machine_configs = contents(machines)
    print('Machines:')
    for config in machine_configs:
        if config.endswith('.cfg'):
            print('   {}'.format(os.path.splitext(config)[0]))


def main():
    parser = argparse.ArgumentParser(
        description='List the available test cases or machines')
    parser.add_argument("-t", "--test_expr", dest="test_expr",
                        help="A regular expression for a test path name to "
                             "search for",
                        metavar="TEST")
    parser.add_argument("-n", "--number", dest="number", type=int,
                        help="The number of the test to list")
    parser.add_argument("--machines", dest="machines", action="store_true",
                        help="List supported machines (instead of test cases)")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="List details of each test case, not just the "
                             "path")
    args = parser.parse_args(sys.argv[2:])
    if args.machines:
        list_machines()
    else:
        list_cases(test_expr=args.test_expr, number=args.number,
                   verbose=args.verbose)
