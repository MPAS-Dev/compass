#!/usr/bin/env python

from __future__ import print_function

import os
import platform
try:
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request

import sys

from shared import parse_args, get_config, get_conda_base, check_call


def bootstrap(activate_install_env, source_path):

    print('Creating the compass conda environment')
    bootstrap_command = '{}/conda/bootstrap.py'.format(source_path)
    command = '{}; ' \
              '{} {}'.format(activate_install_env, bootstrap_command,
                             ' '.join(sys.argv[1:]))
    check_call(command)
    sys.exit(0)


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


def setup_install_env(activate_base):
    print('Setting up a conda environment for installing compass')
    commands = '{}; ' \
               'conda create -y -n temp_compass_install ' \
               'progressbar2 jinja2 mache=1.0.0'.format(activate_base)

    check_call(commands)


def remove_install_env(activate_base):
    print('Removing conda environment for installing compass')
    commands = '{}; ' \
               'conda remove -y --all -n ' \
               'temp_compass_install'.format(activate_base)

    check_call(commands)


def main():
    args = parse_args()
    source_path = os.getcwd()

    config = get_config(args.config_file)

    is_test = not config.getboolean('deploy', 'release')

    conda_base = get_conda_base(args.conda_base, is_test, config)

    base_activation_script = os.path.abspath(
        '{}/etc/profile.d/conda.sh'.format(conda_base))

    activate_base = 'source {}; conda activate'.format(base_activation_script)

    activate_install_env = \
        'source {}; ' \
        'conda activate temp_compass_install'.format(base_activation_script)

    # install miniconda if needed
    install_miniconda(conda_base, activate_base)

    setup_install_env(activate_base)

    bootstrap(activate_install_env, source_path)

    remove_install_env(activate_base)


if __name__ == '__main__':
    main()
