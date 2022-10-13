#!/usr/bin/env python

import argparse
import configparser
import os
import subprocess


def run(launch_path, mpas_path, work_base, load_script, make_command,
        setup_command, run_command):
    """
    This function runs a single step in the bisection process.  It is typically
    called through ``git bisect run`` within the ``utils/bisect/bisect.py`` but
    could be called on its own for testing purposes.

    Parameters
    ----------
    launch_path : str
        The path from which relative paths in the config file are defined,
        typically the root of the compass branch where the config file
        resides and where ``utils/bisect/bisect.py`` was called.
    mpas_path : str
        The relative or absolute path to the mpas component to be built.
    work_base : str
        The base directory for creating work directories for testing the code.
        Subdirectories called ``e3sm_hash<hash>`` will be created with each
        E3SM commit hash that is tested.
    load_script : str
        The relative or absolute path to the load script used to activate
        the compass conda environment and set environment variables used to
        build the MPAS component to test.
    make_command : str
        The command to use to build the MPAS component
    setup_command : str
        The command to use to set up the compass test case(s)
    run_command : str
        The command (typically just ``compass run``) use to run the compass
        test case(s)
    """

    mpas_path = to_abs(mpas_path, launch_path)
    work_base = to_abs(work_base, launch_path)
    load_script = to_abs(load_script, launch_path)

    commands = f'cd {mpas_path}; ' \
               f'source {load_script}; ' \
               f'git rev-parse --short HEAD'
    git_hash = subprocess.check_output(
        commands, shell=True).decode('utf-8').strip('\n')
    git_hash = git_hash.split('\n')[-1]

    work_path = os.path.join(work_base, f'e3sm_hash_{git_hash}')

    try:
        os.makedirs(work_path)
    except FileExistsError:
        pass

    os.chdir(mpas_path)
    commands = f'source {load_script}; ' \
               f'git submodule update --init --recursive; ' \
               f'make clean >& {work_path}/clean.log; ' \
               f'{make_command} >& {work_path}/make.log; ' \
               f'{setup_command} -p {mpas_path} -w {work_path}; ' \
               f'cd {work_path}; ' \
               f'{run_command}'
    print('\n')
    print(72*'-')
    print('Biscect Step')
    print(72*'-')
    print('\nRunning:')
    print_commands = commands.replace('; ', '\n  ')
    print(f'  {print_commands}\n\n')
    subprocess.check_call(commands, shell=True)


def to_abs(path, launch_path):
    """
    Convert a relative path to an absolute path

    Parameters
    ----------
    path : str
        A relative or absolute path
    launch_path : str
        The base path to use to convert relative paths to absolute paths

    Returns
    -------
    path : str
        The original ``path`` as an absolute path
    """
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(launch_path, path))
    return path


def main():
    parser = argparse.ArgumentParser(
        description='Used internally by "git bisect run" to find the first '
                    'E3SM commit for which a given test fails')
    parser.add_argument("-f", "--config_file", dest="config_file",
                        required=True,
                        help="Configuration file with bisect options",
                        metavar="FILE")

    args = parser.parse_args()

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    config.read(args.config_file)

    launch_path = os.path.dirname(args.config_file)

    section = config['bisect']
    run(launch_path=launch_path, mpas_path=section['mpas_path'],
        work_base=section['work_base'], load_script=section['load_script'],
        make_command=section['make_command'],
        setup_command=section['setup_command'],
        run_command=section['run_command'])
    print ('\n')


if __name__ == '__main__':
    main()
