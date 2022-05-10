import argparse
import os
import subprocess
import sys

from compass.version import __version__ as compass_version
from jinja2 import Template
from importlib import resources


def get_conda_base_and_env():
    if 'CONDA_EXE' in os.environ:
        conda_exe = os.environ['CONDA_EXE']
        conda_base = os.path.abspath(
            os.path.join(conda_exe, '..', '..'))
    else:
        raise ValueError('No conda executable detected.')

    if 'CONDA_DEFAULT_ENV' in os.environ:
        conda_env = os.environ['CONDA_DEFAULT_ENV']
    else:
        raise ValueError('No conda environment detected.')

    return conda_base, conda_env


def get_mpi():

    for mpi in ['mpich', 'openmpi']:
        check = subprocess.check_output(
            ['conda', 'list', 'mpich']).decode('utf-8')
        if mpi in check:
            return mpi

    return None


def main():
    parser = argparse.ArgumentParser(
        description='Generate a load script for a Linux or OSX machine')
    parser.parse_args()

    conda_base, conda_env = get_conda_base_and_env()

    mpi = get_mpi()

    if mpi is None:
        suffix = ''
    else:
        suffix = f'_{mpi}'

    if sys.platform == 'Linux':
        env_vars = 'export MPAS_EXTERNAL_LIBS="-lgomp"'
    else:
        env_vars = ''

    script_filename = f'load_compass_{compass_version}{suffix}.sh'
    script_filename = os.path.abspath(script_filename)

    template = Template(resources.read_text(
        'compass.load', 'load_script.template'))
    text = template.render(conda_base=conda_base, conda_env=conda_env,
                           env_vars=env_vars, load_script=script_filename)

    with open(script_filename, 'w') as handle:
        handle.write(text)
