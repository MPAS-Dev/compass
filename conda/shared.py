from __future__ import print_function

import os
import warnings
import sys
import argparse
import subprocess
import platform

try:
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request


def parse_args(bootstrap):
    parser = argparse.ArgumentParser(
        description='Deploy a compass conda environment')
    parser.add_argument("-m", "--machine", dest="machine",
                        help="The name of the machine for loading machine-"
                             "related config options")
    parser.add_argument("--conda", dest="conda_base",
                        help="Path to the conda base")
    parser.add_argument("--spack", dest="spack_base",
                        help="Path to the spack base")
    parser.add_argument("--env_name", dest="env_name",
                        help="The conda environment name and activation script"
                             " prefix")
    parser.add_argument("-p", "--python", dest="python", type=str,
                        help="The python version to deploy")
    parser.add_argument("-i", "--mpi", dest="mpi", type=str,
                        help="The MPI library to deploy, see the docs for "
                             "details")
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
    parser.add_argument("--use_local", dest="use_local", action='store_true',
                        help="Use locally built conda packages (for testing).")
    parser.add_argument("--update_spack", dest="update_spack",
                        action='store_true',
                        help="If the shared spack environment should be "
                             "created or recreated.")
    parser.add_argument("--tmpdir", dest="tmpdir",
                        help="A temporary directory for building spack "
                             "packages")
    parser.add_argument("--with_albany", dest="with_albany",
                        action='store_true',
                        help="Whether to include albany in the spack "
                             "environment")
    parser.add_argument("--without_openmp", dest="without_openmp",
                        action='store_true',
                        help="If this flag is included, OPENMP=true will not "
                             "be added to the load script.  By default, MPAS "
                             "builds will be with OpenMP.")
    if bootstrap:
        parser.add_argument("--local_conda_build", dest="local_conda_build",
                            type=str,
                            help="A path for conda packages (for testing).")

    args = parser.parse_args(sys.argv[1:])

    return args


def get_conda_base(conda_base, config, shared=False):
    if shared:
        conda_base = config.get('paths', 'compass_envs')
    elif conda_base is None:
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
    # handle "~" in the path
    conda_base = os.path.abspath(os.path.expanduser(conda_base))
    return conda_base


def get_spack_base(spack_base, config):
    if spack_base is None:
        if config.has_option('deploy', 'spack'):
            spack_base = config.get('deploy', 'spack')
        else:
            raise ValueError('No spack base provided with --spack and none is '
                             'provided in a config file.')
    # handle "~" in the path
    spack_base = os.path.abspath(os.path.expanduser(spack_base))
    return spack_base


def check_call(commands, env=None):
    print('running: {}'.format(commands))
    proc = subprocess.Popen(commands, env=env, executable='/bin/bash',
                            shell=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, commands)


def install_miniconda(conda_base, activate_base):
    if not os.path.exists(conda_base):
        print('Installing Miniconda3')
        if platform.system() == 'Linux':
            system = 'Linux'
        elif platform.system() == 'Darwin':
            system = 'MacOSX'
        else:
            system = 'Linux'
        miniconda = 'Miniconda3-latest-{}-x86_64.sh'.format(system)
        url = 'https://repo.continuum.io/miniconda/{}'.format(miniconda)
        print(url)
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        f = urlopen(req)
        html = f.read()
        with open(miniconda, 'wb') as outfile:
            outfile.write(html)
        f.close()

        command = '/bin/bash {} -b -p {}'.format(miniconda, conda_base)
        check_call(command)
        os.remove(miniconda)

    print('Doing initial setup')
    commands = '{}; ' \
               'conda config --add channels conda-forge; ' \
               'conda config --set channel_priority strict; ' \
               'conda install -y mamba boa; ' \
               'conda update -y --all'.format(activate_base)

    check_call(commands)
