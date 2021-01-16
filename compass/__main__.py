#!/usr/bin/env python

import sys
import argparse

import compass
from compass.list import main as main_list
from compass.setup import main as main_setup
from compass.clean import main as main_clean
from compass.suite import main as main_suite


def main():
    """
    Entry point for the main script ``compass``
    """

    parser = argparse.ArgumentParser(
        description="Perform compass operations",
        usage='''
compass <command> [<args>]

The available compass commands are:
    list    List the available test cases
    setup   Set up a test case
    clean   Clean up a test case
    suite   Manage a regression test suite

 To get help on an individual command, run:

    compass <command> --help
    ''')

    parser.add_argument('command', help='command to run')
    parser.add_argument('-v', '--version',
                        action='version',
                        version='compass {}'.format(compass.__version__),
                        help="Show version number and exit")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args(sys.argv[1:2])

    commands = {'list': main_list,
                'setup': main_setup,
                'clean': main_clean,
                'suite': main_suite}
    if args.command not in commands:
        print('Unrecognized command {}'.format(args.command))
        parser.print_help()
        exit(1)

    # call the function associated with the requested command
    commands[args.command]()


if __name__ == "__main__":
    main()
