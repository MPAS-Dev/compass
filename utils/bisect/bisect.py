#!/usr/bin/env python

import argparse
import configparser
import os
import subprocess


def bisect(good, bad, e3sm_path, load_script, config_file, first_parent):
    """
    The driver function for calling ``git bisect`` to find the first "bad"
    commit between a known "good" and "bad" E3SM commit.

    The function uses ``git bisect run`` to call
    ``utils/bisect/bisect_step.py`` repeatedly to test whether a given commit
    is good or bad.

    Parameters
    ----------
    good : str
        The hash or tag of a "good" E3SM commit that passes the test(s)
        specified in the config file
    bad : str
        The hash or tag of a "bad" E3SM commit that fails the test(s)
    e3sm_path : str
        The path to the E3SM branch to test.  If you are pointing to the
        ``E3SM-Project`` or ``MALI-Dev`` submodules, make sure they have been
        initialized with ``git submodule update --init``.
    load_script : str
        The relative or absolute path to the load script used to activate
        the compass conda environment and set environment variables used to
        build the MPAS component to test.
    config_file : str
        The relative or absolute path to a config file containing config
        options similar to ``utils/bisect/example.cfg`` that control the
        bisection process.
    first_parent : bool
        Whether to only follow the first parent for merge commits.  This is
        typically desirable because there may be broken commits within a branch
        that are fixed by the time the branch is merged.
    """

    e3sm_path = os.path.abspath(e3sm_path)
    load_script = os.path.abspath(load_script)
    config_file = os.path.abspath(config_file)

    cwd = os.getcwd()

    if first_parent:
        flags = '--first-parent'
    else:
        flags = ''

    commands = f'source {load_script}; ' \
               f'cd {e3sm_path}; ' \
               f'git bisect start {flags}; ' \
               f'git bisect good {good}; ' \
               f'git bisect bad {bad}; ' \
               f'git bisect run {cwd}/utils/bisect/bisect_step.py' \
               f'  -f {config_file}'
    print('\nRunning:')
    print_commands = commands.replace('; ', '\n  ')
    print(f'  {print_commands}\n\n')
    subprocess.check_call(commands, shell=True)


def main():
    parser = argparse.ArgumentParser(
        description='Use "git bisect" to find the first E3SM commit for which '
                    'a given test fails')
    parser.add_argument("-f", "--config_file", dest="config_file",
                        required=True,
                        help="Configuration file with bisect options",
                        metavar="FILE")

    args = parser.parse_args()

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    config.read(args.config_file)

    section = config['bisect']

    bisect(good=section['good'], bad=section['bad'],
           e3sm_path=section['e3sm_path'],
           load_script=section['load_script'],
           config_file=args.config_file,
           first_parent=section.getboolean('first_parent'))


if __name__ == '__main__':
    main()
