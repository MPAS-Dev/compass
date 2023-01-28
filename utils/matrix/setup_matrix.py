#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import shutil

try:
    from configparser import ConfigParser
except ImportError:
    from six.moves import configparser
    import six

    if six.PY2:
        ConfigParser = configparser.SafeConfigParser
    else:
        ConfigParser = configparser.ConfigParser

from shared import get_logger, check_call


# build targets from
# https://mpas-dev.github.io/compass/latest/developers_guide/machines/index.html#supported-machines
all_build_targets = {
    'anvil': {
        ('intel', 'impi'): 'intel-mpi',
        ('intel', 'openmpi'): 'ifort',
        ('intel', 'mvapich'): 'ifort',
        ('gnu', 'openmpi'): 'gfortran',
        ('gnu', 'mvapich'): 'gfortran'},
    'chicoma-cpu': {
        ('gnu', 'mpich'): 'gnu-cray'},
    'chrysalis': {
        ('intel', 'impi'): 'intel-mpi',
        ('intel', 'openmpi'): 'ifort',
        ('gnu', 'openmpi'): 'gfortran'},
    'compy': {
        ('intel', 'impi'): 'intel-mpi',
        ('gnu', 'openmpi'): 'gfortran'},
    'cori-haswell': {
        ('intel', 'mpt'): 'intel-cray',
        ('gnu', 'mpt'): 'gnu-cray'},
    'pm-cpu': {
        ('gnu', 'mpich'): 'gnu-cray'},
    'conda-linux': {
        ('gfortran', 'mpich'): 'gfortran',
        ('gfortran', 'openmpi'): 'gfortran'},
    'conda-osx': {
        ('gfortran-clang', 'mpich'): 'gfortran-clang',
        ('gfortran-clang', 'openmpi'): 'gfortran-clang'}
}


def setup_matrix(config_filename, submit):
    """
    Build and set up (and optionally submit jobs for) a matrix of MPAS builds

    Parameters
    ----------
    config_filename : str
        The name of the config file containing config options to use for both
        the matrix and the test case(s)

    submit : bool
        Whether to submit each suite or set of tests once it has been built
        and set up
    """

    config = ConfigParser()
    config.read(config_filename)

    matrix_filename = 'conda/logs/matrix.log'
    if not os.path.exists(matrix_filename):
        raise OSError(
            '{} not found.  Try running ./conda/config_compass_env.py to '
            'generate it.'.format(matrix_filename))
    with open(matrix_filename, 'r') as f:
        machine = f.readline().strip()
        lines = f.readlines()
        compilers = list()
        mpis = list()
        for line in lines:
            compiler, mpi = line.split(',')
            compilers.append(compiler.strip())
            mpis.append(mpi.strip())

    if machine not in all_build_targets:
        raise ValueError('build targets not known for machine: '
                         '{}'.format(machine))
    build_targets = all_build_targets[machine]

    env_name = config.get('matrix', 'env_name')

    openmp = config.get('matrix', 'openmp')
    openmp = [value.strip().lower() == 'true' for value in
              openmp.replace(',', '').split(' ')]
    debug = config.get('matrix', 'debug')
    debug = [value.strip().lower() == 'true' for value in
             debug.replace(',', '').split(' ')]
    other_build_flags = config.get('matrix', 'other_build_flags')

    mpas_path = config.get('matrix', 'mpas_path')
    mpas_path = os.path.abspath(mpas_path)

    setup_command = config.get('matrix', 'setup_command')
    work_base = config.get('matrix', 'work_base')
    work_base = os.path.abspath(work_base)
    baseline_base = config.get('matrix', 'baseline_base')
    if baseline_base != '':
        baseline_base = os.path.abspath(baseline_base)

    for compiler, mpi in zip(compilers, mpis):
        if (compiler, mpi) not in build_targets:
            raise ValueError('Unsupported compiler {} and MPI '
                             '{}'.format(compiler, mpi))
        target = build_targets[(compiler, mpi)]

        script_name = get_load_script_name(machine, compiler, mpi, env_name)
        script_name = os.path.abspath(script_name)

        for use_openmp in openmp:
            for use_debug in debug:
                suffix = '{}_{}_{}'.format(machine, compiler, mpi)
                make_command = 'make clean; make {} ' \
                               '{}'.format(target, other_build_flags)
                if use_openmp:
                    make_command = '{} OPENMP=true'.format(make_command)
                else:
                    suffix = '{}_noopenmp'.format(suffix)
                    make_command = '{} OPENMP=false'.format(make_command)
                if use_debug:
                    suffix = '{}_debug'.format(suffix)
                    make_command = '{} DEBUG=true'.format(make_command)
                else:
                    make_command = '{} DEBUG=false'.format(make_command)
                mpas_model = build_mpas(
                    script_name, mpas_path, make_command, suffix)

                compass_setup(script_name, setup_command, mpas_path,
                              mpas_model, work_base, baseline_base, config,
                              env_name, suffix, submit)


def get_load_script_name(machine, compiler, mpi, env_name):
    """
    Get the load script for this configuration

    Parameters
    ----------
    machine : str
        The name of the current machine

    compiler : str
        The name of the compiler

    mpi : str
        the MPI library

    env_name : str
        The name of the conda environment to run compass from

    Returns
    -------
    script_name : str
        The name of the load script to source to get the appropriate compass
        environment


    """
    if machine.startswith('conda'):
        script_name = 'load_{}_{}.sh'.format(env_name, mpi)
    else:
        script_name = 'load_{}_{}_{}_{}.sh'.format(env_name, machine,
                                                   compiler, mpi)
    return script_name


def build_mpas(script_name, mpas_path, make_command, suffix):
    """
    Build the MPAS component


    Parameters
    ----------
    script_name : str
        The name of the load script to source to get the appropriate compass
        environment

    mpas_path : str
        The path to the MPAS component to run

    make_command : str
        The make command to run to build the MPAS component

    suffix : str
        A suffix related to the machine, compilers, MPI libraries, etc.

    Returns
    -------
    new_mpas_model : str
        The new name for the MPAS executable with a suffix for the build

    """

    mpas_subdir = os.path.basename(mpas_path)
    if mpas_subdir == 'mpas-ocean':
        mpas_model = 'ocean_model'
    elif mpas_subdir == 'mpas-albany-landice':
        mpas_model = 'landice_model'
    else:
        raise ValueError('Unexpected model subdirectory '
                         '{}'.format(mpas_subdir))

    cwd = os.getcwd()
    print('Changing directory to:\n{}\n'.format(mpas_path))
    os.chdir(mpas_path)
    args = 'source {}; {}'.format(script_name, make_command)

    log_filename = 'build_{}.log'.format(suffix)
    print('\nRunning:\n{}\n'.format('\n'.join(args.split('; '))))
    logger = get_logger(name=__name__, log_filename=log_filename)
    check_call(args, logger=logger)

    new_mpas_model = '{}_{}'.format(mpas_model, suffix)
    shutil.move(mpas_model, new_mpas_model)

    print('Changing directory to:\n{}\n'.format(cwd))
    os.chdir(cwd)

    return new_mpas_model


def compass_setup(script_name, setup_command, mpas_path, mpas_model, work_base,
                  baseline_base, config, env_name, suffix, submit):
    """
    Set up the compass suite or test case(s)

    Parameters
    ----------
    script_name : str
        The name of the load script to source to get the appropriate compass
        environment

    setup_command : str
        The command for setting up the compass test case or suite

    mpas_path : str
        The path to the MPAS component to run

    mpas_model : str
        The name of the MPAS executable within the ``mpas_path``

    work_base : str
        The base work directory for the matrix.  The work directory used for
        the suite or test case(s) is a subdirectory ``suffix`` within this
        directory.

    baseline_base : str
        The base work directory for a baseline matrix to compare to (or an
        empty string for no baseline)

    config : configparser.ConfigParser
        Config options for both the matrix and the test case(s)

    env_name : str
        The name of the conda environment to run compass from

    suffix : str
        A suffix related to the machine, compilers, MPI libraries, etc.

    submit : bool
        Whether to submit each suite or set of tests once it has been built
        and set up
    """

    if not config.has_section('paths'):
        config.add_section('paths')
    config.set('paths', 'mpas_model', mpas_path)
    if not config.has_section('executables'):
        config.add_section('executables')
    config.set('executables', 'model',
               '${{paths:mpas_model}}/{}'.format(mpas_model))

    new_config_filename = '{}_{}.cfg'.format(env_name, suffix)
    with open(new_config_filename, 'w') as f:
        config.write(f)

    work_dir = '{}/{}'.format(work_base, suffix)

    args = 'export NO_COMPASS_REINSTALL=true;' \
           'source {}; ' \
           '{} ' \
           '-p {} ' \
           '-w {} ' \
           '-f {}'.format(script_name, setup_command, mpas_path, work_dir,
                          new_config_filename)

    if baseline_base != '':
        args = '{} -b {}/{}'.format(args, baseline_base, suffix)

    log_filename = 'setup_{}_{}.log'.format(env_name, suffix)
    print('\nRunning:\n{}\n'.format('\n'.join(args.split('; '))))
    logger = get_logger(name=__name__, log_filename=log_filename)
    check_call(args, logger=logger)

    if submit:
        suite = None
        if setup_command.startswith('compass suite'):
            parts = setup_command.split()
            index = parts.index('-t')
            if index == -1:
                index = parts.index('--test_suite')

            if index != -1 and len(parts) > index+1:
                suite = parts[index+1]
        elif setup_command.startswith('compass setup'):
            suite = 'custom'

        if suite is not None:
            job_script = 'job_script.{}.sh'.format(suite)
            if not os.path.exists(os.path.join(work_dir, job_script)):
                raise OSError('Could not find job script {} for suite '
                              '{}'.format(job_script, suite))
            args = 'cd {}; ' \
                   'sbatch {}'.format(work_dir, job_script)
            print('\nRunning:\n{}\n'.format('\n'.join(args.split('; '))))
            check_call(args)


def main():
    parser = argparse.ArgumentParser(
        description='Build MPAS and set up compass with a matrix of build '
                    'configs.')
    parser.add_argument("-f", "--config_file", dest="config_file",
                        required=True,
                        help="Configuration file with matrix build options",
                        metavar="FILE")
    parser.add_argument("--submit", dest="submit", action='store_true',
                        help="Whether to submit the job scripts for each test"
                             "once setup is complete.")

    args = parser.parse_args()
    setup_matrix(args.config_file, args.submit)


if __name__ == '__main__':
    main()
