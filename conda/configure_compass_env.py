#!/usr/bin/env python3

import os
import sys
from configparser import ConfigParser

from shared import (
    check_call,
    get_conda_base,
    get_logger,
    install_miniforge,
    parse_args,
)


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
    bootstrap_command = f'{source_path}/conda/bootstrap.py'
    command = f'{activate_install_env} && ' \
              f'{bootstrap_command} {" ".join(sys.argv[1:])}'
    if local_conda_build is not None:
        command = f'{command} --local_conda_build {local_conda_build}'
    check_call(command)


def setup_install_env(env_name, activate_base, use_local, logger, recreate,
                      conda_base, mache):
    env_path = os.path.join(conda_base, 'envs', env_name)
    if use_local:
        channels = '--use-local'
    else:
        channels = ''
    packages = f'jinja2 {mache} packaging progressbar2'
    if recreate or not os.path.exists(env_path):
        print('Setting up a conda environment for installing compass\n')
        commands = f'{activate_base} && ' \
                   f'conda create -y -n {env_name} {channels} {packages}'
    else:
        print('Updating conda environment for installing compass\n')
        commands = f'{activate_base} && ' \
                   f'conda install -y -n {env_name} {channels} {packages}'

    check_call(commands, logger=logger)


def main():
    args = parse_args(bootstrap=False)
    source_path = os.getcwd()

    if args.tmpdir is not None:
        try:
            os.makedirs(args.tmpdir)
        except FileExistsError:
            pass

    config = get_config(args.config_file)

    conda_base = get_conda_base(args.conda_base, config, warn=True)
    conda_base = os.path.abspath(conda_base)

    env_name = 'compass_bootstrap'

    source_activation_scripts = \
        f'source {conda_base}/etc/profile.d/conda.sh'

    activate_base = f'{source_activation_scripts} && conda activate'

    activate_install_env = \
        f'{source_activation_scripts} && ' \
        f'conda activate {env_name}'
    try:
        os.makedirs('conda/logs')
    except OSError:
        pass

    if args.verbose:
        logger = None
    else:
        logger = get_logger(log_filename='conda/logs/prebootstrap.log',
                            name=__name__)

    # install miniforge if needed
    install_miniforge(conda_base, activate_base, logger)

    local_mache = args.mache_fork is not None and args.mache_branch is not None
    if local_mache:
        mache = ''
    else:
        mache = '"mache=1.17.0"'

    setup_install_env(env_name, activate_base, args.use_local, logger,
                      args.recreate, conda_base, mache)

    if local_mache:
        print('Clone and install local mache\n')
        commands = f'{activate_install_env} && ' \
                   f'rm -rf conda/build_mache && ' \
                   f'mkdir -p conda/build_mache && ' \
                   f'cd conda/build_mache && ' \
                   f'git clone -b {args.mache_branch} ' \
                   f'git@github.com:{args.mache_fork}.git mache && ' \
                   f'cd mache && ' \
                   f'python -m pip install .'

        check_call(commands, logger=logger)

    env_type = config.get('deploy', 'env_type')
    if env_type not in ['dev', 'test_release', 'release']:
        raise ValueError(f'Unexpected env_type: {env_type}')

    if env_type == 'test_release' and args.use_local:
        local_conda_build = os.path.abspath(f'{conda_base}/conda-bld')
    else:
        local_conda_build = None

    bootstrap(activate_install_env, source_path, local_conda_build)


if __name__ == '__main__':
    main()
