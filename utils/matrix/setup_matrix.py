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
    'badger': {
        ('intel', 'impi'): 'intel-mpi',
        ('gnu', 'mvapich'): 'gfortran'},
    'chrysalis': {
        ('intel', 'impi'): 'intel-mpi',
        ('intel', 'openmpi'): 'ifort',
        ('gnu', 'openmpi'): 'gfortran'},
    'compy': {
        ('intel', 'impi'): 'intel-mpi'},
    'cori-haswell': {
        ('intel', 'mpt'): 'intel-nersc',
        ('gnu', 'mpt'): 'gnu-nersc'},
    'conda-linux': {
        ('gfortran', 'mpich'): 'gfortran',
        ('gfortran', 'openmpi'): 'gfortran'},
    'conda-osx': {
        ('gfortran-clang', 'mpich'): 'gfortran-clang',
        ('gfortran-clang', 'openmpi'): 'gfortran-clang'}
}


def setup_matrix(config_filename):

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
                              env_name, suffix)


def get_load_script_name(machine, compiler, mpi, env_name):
    if machine.startswith('conda'):
        script_name = 'load_{}_{}.sh'.format(env_name, mpi)
    else:
        script_name = 'load_{}_{}_{}_{}.sh'.format(env_name, machine,
                                                   compiler, mpi)
    return script_name


def build_mpas(script_name, mpas_path, make_command, suffix):

    mpas_subdir = os.path.basename(mpas_path)
    if mpas_subdir == 'mpas-ocean':
        mpas_model = 'ocean_model'
    elif mpas_subdir == 'mpas-albany-landice':
        mpas_model = 'landice_model'
    else:
        raise ValueError('Unexpected model subdirectory '
                         '{}'.format(mpas_subdir))

    cwd = os.getcwd()
    os.chdir(mpas_path)
    args = 'source {}; {}'.format(script_name, make_command)

    log_filename = 'build_{}.log'.format(suffix)
    print('\nRunning:\n{}\n'.format('\n'.join(args.split('; '))))
    logger = get_logger(name=__name__, log_filename=log_filename)
    check_call(args, logger=logger)

    new_mpas_model = '{}_{}'.format(mpas_model, suffix)
    shutil.move(mpas_model, new_mpas_model)

    os.chdir(cwd)

    return new_mpas_model


def compass_setup(script_name, setup_command, mpas_path, mpas_model, work_base,
                  baseline_base, config, env_name, suffix):

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

    args = 'source {}; ' \
           '{} ' \
           '-p {} ' \
           '-w {}/{} ' \
           '-f {}'.format(script_name, setup_command, mpas_path, work_base,
                          suffix, new_config_filename)

    if baseline_base != '':
        args = '{} -b {}/{}'.format(args, baseline_base, suffix)

    log_filename = 'setup_{}_{}.log'.format(env_name, suffix)
    print('\nRunning:\n{}\n'.format('\n'.join(args.split('; '))))
    logger = get_logger(name=__name__, log_filename=log_filename)
    check_call(args, logger=logger)


def main():
    parser = argparse.ArgumentParser(
        description='Build MPAS and set up compass with a matrix of build '
                    'configs.')
    parser.add_argument("-f", "--config_file", dest="config_file",
                        required=True,
                        help="Configuration file with matrix build options",
                        metavar="FILE")

    args = parser.parse_args()
    setup_matrix(args.config_file)


if __name__ == '__main__':
    main()
