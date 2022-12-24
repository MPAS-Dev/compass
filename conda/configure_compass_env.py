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

from shared import parse_args, get_conda_base, check_call, install_miniconda, \
    get_logger


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

    print('Creating the compass conda environment\n')
    bootstrap_command = '{}/conda/bootstrap.py'.format(source_path)
    command = '{}; ' \
              '{} {}'.format(activate_install_env, bootstrap_command,
                             ' '.join(sys.argv[1:]))
    if local_conda_build is not None:
        command = '{} --local_conda_build {}'.format(command,
                                                     local_conda_build)
    check_call(command)


def setup_install_env(env_name, activate_base, use_local, logger, recreate,
                      conda_base):
    env_path = os.path.join(conda_base, 'envs', env_name)
    if use_local:
        channels = '--use-local'
    else:
        channels = ''
    packages = 'progressbar2 jinja2 "mache=1.10.0"'
    if recreate or not os.path.exists(env_path):
        print('Setting up a conda environment for installing compass\n')
        commands = '{}; ' \
                   'mamba create -y -n {} {} {}'.format(activate_base,
                                                        env_name, channels,
                                                        packages)
    else:
        print('Updating conda environment for installing compass\n')
        commands = '{}; ' \
                   'mamba install -y -n {} {} {}'.format(activate_base,
                                                         env_name, channels,
                                                         packages)

    check_call(commands, logger=logger)


def main():
    args = parse_args(bootstrap=False)
    source_path = os.getcwd()

    config = get_config(args.config_file)

    conda_base = get_conda_base(args.conda_base, config, warn=True)
    conda_base = os.path.abspath(conda_base)

    env_name = 'compass_bootstrap'

    source_activation_scripts = \
        'source {}/etc/profile.d/conda.sh; ' \
        'source {}/etc/profile.d/mamba.sh'.format(conda_base, conda_base)

    activate_base = '{}; conda activate'.format(source_activation_scripts)

    activate_install_env = \
        '{}; ' \
        'conda activate {}'.format(source_activation_scripts, env_name)
    try:
        os.makedirs('conda/logs')
    except OSError:
        pass

    logger = get_logger(log_filename='conda/logs/prebootstrap.log',
                        name=__name__)

    # install miniconda if needed
    install_miniconda(conda_base, activate_base, logger)

    setup_install_env(env_name, activate_base, args.use_local, logger,
                      args.recreate, conda_base)

    env_type = config.get('deploy', 'env_type')
    if env_type not in ['dev', 'test_release', 'release']:
        raise ValueError('Unexpected env_type: {}'.format(env_type))

    if env_type == 'test_release' and args.use_local:
        local_conda_build = os.path.abspath('{}/conda-bld'.format(conda_base))
    else:
        local_conda_build = None

    bootstrap(activate_install_env, source_path, local_conda_build)


if __name__ == '__main__':
    main()
