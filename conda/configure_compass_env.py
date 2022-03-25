#!/usr/bin/env python

from __future__ import print_function

import os
import sys

try:
    from configparser import ConfigParser
except ImportError:
    from six.moves import configparser
    import six

    if six.PY2:
        ConfigParser = configparser.SafeConfigParser
    else:
        ConfigParser = configparser.ConfigParser

from shared import parse_args, get_conda_base, check_call, install_miniconda


def get_config(config_file):
    # we can't load compass so we find the config files
    here = os.path.abspath(os.path.dirname(__file__))
    default_config = os.path.join(here, 'default.cfg')
    config = ConfigParser()
    config.read(default_config)

    if config_file is not None:
        config.read(config_file)

    return config


def bootstrap(activate_install_env, source_path, local_conda_build):

    print('Creating the compass conda environment')
    bootstrap_command = '{}/conda/bootstrap.py'.format(source_path)
    command = '{}; ' \
              '{} {}'.format(activate_install_env, bootstrap_command,
                             ' '.join(sys.argv[1:]))
    if local_conda_build is not None:
        command = '{} --local_conda_build {}'.format(command,
                                                     local_conda_build)
    check_call(command)


def setup_install_env(activate_base, use_local):
    if use_local:
        channels = '--use-local'
    else:
        channels = ''
    print('Setting up a conda environment for installing compass')
    commands = '{}; ' \
               'mamba create -y -n temp_compass_install {} ' \
               'progressbar2 jinja2 "mache>=1.3.2"'.format(activate_base,
                                                           channels)

    check_call(commands)


def remove_install_env(activate_base):
    print('Removing conda environment for installing compass')
    commands = '{}; ' \
               'conda remove -y --all -n ' \
               'temp_compass_install'.format(activate_base)

    check_call(commands)


def main():
    args = parse_args(bootstrap=False)
    source_path = os.getcwd()

    config = get_config(args.config_file)

    conda_base = get_conda_base(args.conda_base, config)

    base_activation_script = os.path.abspath(
        '{}/etc/profile.d/conda.sh'.format(conda_base))

    activate_base = 'source {}; conda activate'.format(base_activation_script)

    activate_install_env = \
        'source {}; ' \
        'conda activate temp_compass_install'.format(base_activation_script)

    # install miniconda if needed
    install_miniconda(conda_base, activate_base)

    setup_install_env(activate_base, args.use_local)

    env_type = config.get('deploy', 'env_type')
    if env_type not in ['dev', 'test_release', 'release']:
        raise ValueError('Unexpected env_type: {}'.format(env_type))

    if env_type == 'test_release' and args.use_local:
        local_conda_build = os.path.abspath('{}/conda-bld'.format(conda_base))
    else:
        local_conda_build = None

    bootstrap(activate_install_env, source_path, local_conda_build)

    remove_install_env(activate_base)


if __name__ == '__main__':
    main()
