import os
import warnings
import sys
import argparse
import subprocess
from importlib.resources import path

try:
    from configparser import ConfigParser
except ImportError:
    from six.moves import configparser
    import six

    if six.PY2:
        ConfigParser = configparser.SafeConfigParser
    else:
        ConfigParser = configparser.ConfigParser


def parse_args():
    parser = argparse.ArgumentParser(
        description='Deploy a compass conda environment')
    parser.add_argument("-m", "--machine", dest="machine",
                        help="The name of the machine for loading machine-"
                             "related config options")
    parser.add_argument("--conda", dest="conda_base",
                        help="Path to the conda base")
    parser.add_argument("--env_name", dest="env_name",
                        help="The conda environment name and activation script"
                             " prefix")
    parser.add_argument("-p", "--python", dest="python", type=str,
                        help="The python version to deploy")
    parser.add_argument("-i", "--mpi", dest="mpi", type=str,
                        help="The MPI library (nompi, mpich, openmpi or a "
                             "system flavor) to deploy")
    parser.add_argument("-c", "--compiler", dest="compiler", type=str,
                        help="The name of the compiler")
    parser.add_argument("--env_only", dest="env_only", action='store_true',
                        help="Create only the compass environment, don't "
                             "install compilers or build SCORPIO")
    parser.add_argument("--recreate", dest="recreate", action='store_true',
                        help="Recreate the environment if it exists")
    parser.add_argument("-f", "--config_file", dest="config_file",
                        help="Config file to override deployment config "
                             "options")
    parser.add_argument("--check", dest="check", action='store_true',
                        help="Check the resulting environment for expected "
                             "packages")

    args = parser.parse_args(sys.argv[1:])

    return args


def get_config(config_file, machine=None):
    # we can't load compass so we find the config files
    here = os.path.abspath(os.path.dirname(__file__))
    default_config = os.path.join(here, '..', 'compass', 'default.cfg')
    config = ConfigParser()
    config.read(default_config)

    if machine is not None:
        with path('mache.machines', f'{machine}.cfg') as machine_config:
            config.read(str(machine_config))

        machine_config = os.path.join(here, '..', 'compass', 'machines',
                                      '{}.cfg'.format(machine))
        config.read(machine_config)

    if config_file is not None:
        config.read(config_file)

    return config


def get_conda_base(conda_base, is_test, config):
    if conda_base is None:
        if is_test:
            if 'CONDA_EXE' in os.environ:
                # if this is a test, assume we're the same base as the
                # environment currently active
                conda_exe = os.environ['CONDA_EXE']
                conda_base = os.path.abspath(
                    os.path.join(conda_exe, '..', '..'))
                warnings.warn(
                    '--conda path not supplied.  Using conda installed at '
                    '{}'.format(conda_base))
            else:
                raise ValueError('No conda base provided with --conda and '
                                 'none could be inferred.')
        else:
            conda_base = config.get('paths', 'compass_envs')
    # handle "~" in the path
    conda_base = os.path.abspath(os.path.expanduser(conda_base))
    return conda_base


def check_call(commands, env=None):
    print('running: {}'.format(commands))
    proc = subprocess.Popen(commands, env=env, executable='/bin/bash',
                            shell=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, commands)
