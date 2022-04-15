#!/usr/bin/env python

import sys
import argparse
import os

from compass import list, setup, clean, suite, cache
from compass.version import __version__
import compass.run.serial as run_serial


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
    run     Run a suite, test case or step

 To get help on an individual command, run:

    compass <command> --help
    ''')

    parser.add_argument('command', help='command to run')
    parser.add_argument('-v', '--version',
                        action='version',
                        version=f'compass {__version__}',
                        help="Show version number and exit")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args(sys.argv[1:2])

    # only allow the "compass cache" command if we're on Anvil or Chrysalis
    allow_cache = ('COMPASS_MACHINE' in os.environ and
                   os.environ['COMPASS_MACHINE'] in ['anvil', 'chrysalis'])

    commands = {'list': list.main,
                'setup': setup.main,
                'clean': clean.main,
                'suite': suite.main,
                'run': run_serial.main}
    if allow_cache:
        commands['cache'] = cache.main

    if args.command not in commands:
        print('Unrecognized command {}'.format(args.command))
        parser.print_help()
        exit(1)

    # call the function associated with the requested command
    commands[args.command]()


if __name__ == "__main__":
    main()
