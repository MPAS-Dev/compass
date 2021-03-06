#!/usr/bin/env python

from __future__ import print_function

import os
import argparse
import sys
import re
import socket
import warnings
import glob

try:
    from configparser import ConfigParser
except ImportError:
    from six.moves import configparser
    import six

    if six.PY2:
        ConfigParser = configparser.SafeConfigParser
    else:
        ConfigParser = configparser.ConfigParser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Deploy a compass conda environment')
    parser.add_argument("-m", "--machine", dest="machine",
                        help="The name of the machine")
    parser.add_argument("-c", "--compiler", dest="compiler", type=str,
                        help="The name of the compiler")
    parser.add_argument("-i", "--mpi", dest="mpi", type=str,
                        help="The MPI library")
    parser.add_argument("--list", dest="list", action="store_true",
                        help="list the available compilers and MPI libraries")

    args = parser.parse_args(sys.argv[1:])

    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, '..', 'compass', '__init__.py')) as f:
        init_file = f.read()

    version = re.search(r'{}\s*=\s*[(]([^)]*)[)]'.format('__version_info__'),
                        init_file).group(1).replace(', ', '.')

    default_config = os.path.join(here, '..', 'compass', 'default.cfg')
    config = ConfigParser()
    config.read(default_config)

    if args.machine is not None:
        machine = args.machine
    else:
        hostname = socket.gethostname()
        if hostname.startswith('cori'):
            warnings.warn('defaulting to cori-haswell.  Use -m cori-knl if you'
                          ' wish to run on KNL.')
            machine = 'cori-haswell'
        elif hostname.startswith('blueslogin'):
            machine = 'anvil'
        elif hostname.startswith('chrlogin'):
            machine = 'chrysalis'
        elif hostname.startswith('compy'):
            machine = 'compy'
        elif hostname.startswith('gr-fe'):
            machine = 'grizzly'
        elif hostname.startswith('ba-fe'):
            machine = 'badger'
        else:
            raise ValueError('No machine name was supplied and hostname {} was'
                             ' not recognized.'.format(hostname))

    machine_config = os.path.join(here, '..', 'compass', 'machines',
                                  '{}.cfg'.format(machine))

    if not os.path.exists(machine_config):
        raise ValueError('machine {} doesn\'t have a config file in '
                         'compass/machines.'.format(machine))

    config.read(machine_config)

    base_path = config.get('paths', 'compass_envs')
    activ_path = os.path.abspath(os.path.join(base_path, '..'))

    if args.list:
        files = glob.glob('{}/test_compass{}*.sh'.format(
            activ_path, version))
        compiler = config.get('deploy', 'compiler')
        mpi = config.get('deploy', 'mpi_{}'.format(compiler))
        print('Default compiler and MPI library:')
        print('  -c {} -i {}'.format(compiler, mpi))
        print('')
        print('Available compilers and MPI libraries:')
        for filename in files:
            basename = os.path.splitext(filename)[0].split('_')
            compiler = basename[-2]
            mpi = basename[-1]
            print('  -c {} -i {}'.format(compiler, mpi))
    else:    
        if args.compiler is not None:
            compiler = args.compiler
        else:
            compiler = config.get('deploy', 'compiler')

        if args.mpi is not None:
            mpi = args.mpi
        else:
            mpi = config.get('deploy', 'mpi_{}'.format(compiler))

        suffix = '_{}_{}'.format(compiler, mpi)

        script_filename = '{}/test_compass{}{}.sh'.format(
            activ_path, version, suffix)

        with open('./load/tmp_script_name', 'w') as f:
            f.write(script_filename)
