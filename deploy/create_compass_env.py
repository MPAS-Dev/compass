#!/usr/bin/env python

import subprocess
import os
import glob
import stat
import grp
import requests
import progressbar
import argparse
import sys
import re
import configparser


def check_env(base_path, env_name):
    print("Checking the environment {}".format(env_name))

    activate = 'source {}/etc/profile.d/conda.sh; conda activate {}'.format(
        base_path, env_name)

    imports = ['geometric_features', 'mpas_tools', 'jigsawpy', 'compass']
    for import_name in imports:
        command = '{}; python -c "import {}"'.format(activate, import_name)
        test_command(command, os.environ, import_name)

    commands = [['gpmetis', '--help'],
                ['ffmpeg', '--help'],
                ['compass', 'list'],
                ['compass', 'setup', '--help'],
                ['compass', 'suite', '--help'],
                ['compass', 'clean', '--help']]

    for command in commands:
        package = command[0]
        command = '{}; {}'.format(activate, ' '.join(command))
        test_command(command, os.environ, package)


def test_command(command, env, package):
    try:
        subprocess.check_call(command, env=env, executable='/bin/bash',
                              shell=True)
    except subprocess.CalledProcessError as e:
        print('  {} failed'.format(package))
        raise e
    print('  {} passes'.format(package))


def main():
    parser = argparse.ArgumentParser(
        description='Deploy a compass conda environment', prog='compass-deploy')
    parser.add_argument("-m", "--machine", dest="machine",
                        help="The name of the machine for loading machine-"
                             "related config options", metavar="MACH")
    parser.add_argument("-f", "--config_file", dest="config_file",
                        help="Config file to override deployment config "
                             "options")
    parser.add_argument("-t", "--test", dest="test", type=bool,
                        help="Whether to deploy a test environment")
    parser.add_argument("-b", "--build", dest="build", type=bool,
                        help="Whether to build the test package")
    parser.add_argument("-r", "--recreate", dest="recreate", type=bool,
                        help="Whether to recreate the environment if it "
                             "exists")
    parser.add_argument("-s", "--suffix", dest="suffix", type=str,
                        help="A suffix to append to the environment name")
    parser.add_argument("-p", "--python", dest="python", type=str,
                        help="The python version to deploy")
    parser.add_argument("-i", "--mpi", dest="mpi", type=str,
                        help="The MPI flavor (nompi, mpich, openmpi) to "
                             "deploy")
    parser.add_argument("-g", "--group", dest="group", type=str,
                        help="The unix group that should own the environment")

    args = parser.parse_args(sys.argv[1:])

    # we can't import compass because we probably don't have the necessary
    # dependencies, so we get the version by parsing as in the root setup.py
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, '..', 'compass', '__init__.py')) as f:
        init_file = f.read()

    version = re.search(r'{}\s*=\s*[(]([^)]*)[)]'.format('__version_info__'),
                        init_file).group(1).replace(', ', '.')

    # again, we can't load compass so we find the config files
    default_config = os.path.join(here, '..', 'compass', 'default.cfg')
    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    config.read(default_config)
    if args.machine is not None:
        machine_config = os.path.join(here, '..', 'compass', 'machines',
                                      '{}.cfg'.format(args.machine))
        config.read(machine_config)

    if args.config_file is not None:
        config.read(args.config_file)

    if args.group is not None:
        group = args.group
    else:
        group = config.get('deploy', 'group')

    if args.suffix is not None:
        suffix = args.suffix
    else:
        suffix = config.get('deploy', 'suffix')

    if args.python is not None:
        python = args.python
    else:
        python = config.get('deploy', 'python')

    if args.mpi is not None:
        mpi = args.mpi
    else:
        mpi = config.get('deploy', 'mpi')

    if args.test is not None:
        is_test = args.test
    else:
        is_test = config.getboolean('deploy', 'test')

    if args.build is not None:
        build = args.build
    else:
        build = config.getboolean('deploy', 'build')

    if args.test is not None:
        force_recreate = args.recreate
    else:
        force_recreate = config.getboolean('deploy', 'recreate')

    base_path = config.get('paths', 'compass_envs')
    activ_path = os.path.abspath(os.path.join(base_path, '..'))

    if not os.path.exists(base_path):
        miniconda = 'Miniconda3-latest-Linux-x86_64.sh'
        url = 'https://repo.continuum.io/miniconda/{}'.format(miniconda)
        r = requests.get(url)
        with open(miniconda, 'wb') as outfile:
            outfile.write(r.content)

        command = '/bin/bash {} -b -p {}'.format(miniconda, base_path)
        subprocess.check_call(command, executable='/bin/bash', shell=True)
        os.remove(miniconda)

    activate = 'source {}/etc/profile.d/conda.sh; conda activate'.format(
        base_path)

    print('Doing initial setup')
    commands = '{}; ' \
               'conda config --add channels conda-forge; ' \
               'conda config --set channel_priority strict; ' \
               'conda install -y conda-build; ' \
               'conda update -y --all'.format(activate, base_path)

    subprocess.check_call(commands, executable='/bin/bash', shell=True)
    print('done')

    if is_test and build:
        print('Building the conda package')
        commands = '{}; ' \
                   'conda build -m ci/mpi_{}.yaml recipe'.format(activate, mpi)

        subprocess.check_call(commands, executable='/bin/bash', shell=True)

    if mpi == 'nompi':
        mpi_prefix = 'nompi'
    else:
        mpi_prefix = 'mpi_{}'.format(mpi)

    channels = '--override-channels -c conda-forge -c defaults'
    if is_test:
        channels = '--use-local {}'.format(channels)
    else:
        channels = '{} -c e3sm'.format(channels)
    packages = 'python={} "compass={}={}_*"'.format(
        python, version, mpi_prefix)

    if is_test:
        env_name = 'test_compass_{}{}'.format(version, suffix)
    else:
        env_name = 'compass_{}{}'.format(version, suffix)
    if not os.path.exists('{}/envs/{}'.format(base_path, env_name)) \
            or force_recreate:
        print('creating {}'.format(env_name))
        commands = '{}; conda create -y -n {} {} {}'.format(
            activate, env_name, channels, packages)
        subprocess.check_call(commands, executable='/bin/bash', shell=True)
    else:
        print('{} already exists'.format(env_name))

    check_env(base_path, env_name)

    try:
        os.makedirs(activ_path)
    except FileExistsError:
        pass

    for ext in ['sh', 'csh']:
        script = []
        if ext == 'sh':
            script.extend(['if [ -x "$(command -v module)" ] ; then\n',
                           '  module unload python\n',
                           'fi\n'])
        script.append('source {}/etc/profile.d/conda.{}\n'.format(
            base_path, ext))
        script.append('conda activate {}\n'.format(env_name))

        if is_test:
            file_name = '{}/load_test_compass{}.{}'.format(
                activ_path, suffix, ext)
        else:
            file_name = '{}/load_latest_compass{}.{}'.format(
                activ_path, suffix, ext)
        if os.path.exists(file_name):
            os.remove(file_name)
        with open(file_name, 'w') as f:
            f.writelines(script)

    commands = '{}; conda clean -y -p -t'.format(activate)
    subprocess.check_call(commands, executable='/bin/bash', shell=True)

    print('changing permissions on activation scripts')
    activation_files = glob.glob('{}/load_*_compass*'.format(
        activ_path))

    read_perm = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    exec_perm = (stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR |
                 stat.S_IRGRP | stat.S_IXGRP |
                 stat.S_IROTH | stat.S_IXOTH)

    mask = stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO

    for file_name in activation_files:
        os.chmod(file_name, read_perm)

    print('changing permissions on environments')

    new_uid = os.getuid()
    new_gid = grp.getgrnam(group).gr_gid

    files_and_dirs = []
    for root, dirs, files in os.walk(base_path):
        files_and_dirs.extend(dirs)
        files_and_dirs.extend(files)

    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(),
               ' ', progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets,
                                  maxval=len(files_and_dirs)).start()
    progress = 0
    for root, dirs, files in os.walk(base_path):
        for directory in dirs:
            progress += 1
            bar.update(progress)

            directory = os.path.join(root, directory)

            try:
                dir_stat = os.stat(directory)
            except OSError:
                continue

            perm = dir_stat.st_mode & mask

            if perm == exec_perm and dir_stat.st_uid == new_uid and \
                    dir_stat.st_gid == new_gid:
                continue

            try:
                os.chown(directory, new_uid, new_gid)
                os.chmod(directory, exec_perm)
            except OSError:
                continue

        for file_name in files:
            progress += 1
            bar.update(progress)
            file_name = os.path.join(root, file_name)
            try:
                file_stat = os.stat(file_name)
            except OSError:
                continue

            perm = file_stat.st_mode & mask

            if perm & stat.S_IXUSR:
                # executable, so make sure others can execute it
                new_perm = exec_perm
            else:
                new_perm = read_perm

            if perm == new_perm and file_stat.st_uid == new_uid and \
                    file_stat.st_gid == new_gid:
                continue

            try:
                os.chown(file_name, new_uid, new_gid)
                os.chmod(file_name, perm)
            except OSError:
                continue

    bar.finish()
    print('  done.')


if __name__ == '__main__':
    main()
