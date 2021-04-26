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
from lxml import etree
from jinja2 import Template
import shutil


def get_e3sm_compiler_and_mpi(machine, compiler, mpilib):

    for filename in ['config_machines.xml', 'config_compilers.xml']:
        url = 'https://raw.githubusercontent.com/E3SM-Project/E3SM/master/' \
              'cime_config/machines/{}'.format(filename)
        r = requests.get(url)
        with open(filename, 'wb') as outfile:
            outfile.write(r.content)

    root = etree.parse('config_machines.xml')

    machines = next(root.iter('config_machines'))

    mach = None
    for mach in machines:
        if mach.tag == 'machine' and mach.attrib['MACH'] == machine:
            break

    if mach is None:
        raise ValueError('{} does not appear to be an E3SM supported machine. '
                         'compass cannot be deployed with system compilers.')

    compilers = None
    for child in mach:
        if child.tag == 'COMPILERS':
            compilers = child.text.split(',')
            break

    if compiler not in compilers:
        raise ValueError('Compiler {} not found on {}. Try: {}'.format(
            compiler, machine, compilers))

    mpilibs = None
    for child in mach:
        if child.tag == 'MPILIBS':
            mpilibs = child.text.split(',')
            break

    if mpilib not in mpilibs:
        raise ValueError('MPI library {} not found on {}. Try: {}'.format(
            mpilib, machine, mpilibs))

    machine_os = None
    for child in mach:
        if child.tag == 'OS':
            machine_os = child.text
            break

    commands = []
    modules = next(mach.iter('module_system'))
    for module in modules:
        if module.tag == 'modules':
            include = True
            if 'compiler' in module.attrib and \
                    module.attrib['compiler'] != compiler:
                include = False
            if 'mpilib' in module.attrib and \
                    module.attrib['mpilib'] != mpilib and \
                    module.attrib['mpilib'] != '!mpi-serial':
                include = False
            if include:
                for command in module:
                    if command.tag == 'command':
                        text = 'module {}'.format(command.attrib['name'])
                        if command.text is not None:
                            text = '{} {}'.format(text, command.text)
                        commands.append(text)

    root = etree.parse('config_compilers.xml')

    compilers = next(root.iter('config_compilers'))

    mpicc = None
    mpifc = None
    mpicxx = None
    for comp in compilers:
        if comp.tag != 'compiler':
            continue
        if 'COMPILER' in comp.attrib and comp.attrib['COMPILER'] != compiler:
            continue
        if 'OS' in comp.attrib and comp.attrib['OS'] != machine_os:
            continue
        if 'MACH' in comp.attrib and comp.attrib['MACH'] != machine:
            continue

        # okay, this is either a "generic" compiler section or one for this
        # machine

        for child in comp:
            if 'MPILIB' in child.attrib:
                mpi = child.attrib['MPILIB']
                if mpi[0] == '!':
                    mpi_match = mpi[1:] != mpilib
                else:
                    mpi_match = mpi == mpilib
            else:
                mpi_match = True

            if not mpi_match:
                continue

            if child.tag == 'MPICC':
                mpicc = child.text.strip()
            elif child.tag == 'MPICXX':
                mpicxx = child.text.strip()
            elif child.tag == 'MPIFC':
                mpifc = child.text.strip()

    env_vars = []

    if 'intel' in compiler:
        esmf_compilers = '    export ESMF_COMPILER=intel'
    elif compiler == 'pgi':
        esmf_compilers = '    export ESMF_COMPILER=pgi\n' \
                         '    export ESMF_F90={}\n' \
                         '    export ESMF_CXX={}'.format(mpifc, mpicxx)
    else:
        esmf_compilers = '    export ESMF_F90={}\n' \
                         '    export ESMF_CXX={}'.format(mpifc, mpicxx)

    if 'intel' in compiler and machine == 'anvil':
        env_vars.extend(['export I_MPI_CC=icc',
                         'export I_MPI_CXX=icpc',
                         'export I_MPI_F77=ifort',
                         'export I_MPI_F90=ifort'])

    if mpilib == 'mvapich':
        esmf_comm = 'mvapich2'
        env_vars.extend(['export MV2_ENABLE_AFFINITY=0',
                         'export MV2_SHOW_CPU_BINDING=1'])
    elif mpilib == 'mpich':
        esmf_comm = 'mpich3'
    elif mpilib == 'impi':
        esmf_comm = 'intelmpi'
    else:
        esmf_comm = mpilib

    if machine == 'grizzly':
        esmf_netcdf = \
            '    export ESMF_NETCDF="split"\n' \
            '    export ESMF_NETCDF_INCLUDE=$NETCDF_C_PATH/include\n' \
            '    export ESMF_NETCDF_LIBPATH=$NETCDF_C_PATH/lib64'
    elif machine == 'badger':
        esmf_netcdf = \
            '    export ESMF_NETCDF="split"\n' \
            '    export ESMF_NETCDF_INCLUDE=$NETCDF_C_PATH/include\n' \
            '    export ESMF_NETCDF_LIBPATH=$NETCDF_C_PATH/lib64'
    else:
        esmf_netcdf = '    export ESMF_NETCDF="nc-config"'

    if 'cori' in machine:
        netcdf_paths = 'export NETCDF_C_PATH=$NETCDF_DIR\n' \
                       'export NETCDF_FORTRAN_PATH=$NETCDF_DIR\n' \
                       'export PNETCDF_PATH=$PNETCDF_DIR'
        mpas_netcdf_paths = 'export NETCDF=$NETCDF_DIR\n' \
                            'export NETCDFF=$NETCDF_DIR\n' \
                            'export PNETCDF=$PNETCDF_DIR'
    else:
        netcdf_paths = \
            'export NETCDF_C_PATH=$(dirname $(dirname $(which nc-config)))\n' \
            'export NETCDF_FORTRAN_PATH=$(dirname $(dirname $(which nf-config)))\n' \
            'export PNETCDF_PATH=$(dirname $(dirname $(which pnetcdf-config)))'
        mpas_netcdf_paths = \
            'export NETCDF=$(dirname $(dirname $(which nc-config)))\n' \
            'export NETCDFF=$(dirname $(dirname $(which nf-config)))\n' \
            'export PNETCDF=$(dirname $(dirname $(which pnetcdf-config)))'

    sys_info = dict(modules=commands, mpicc=mpicc, mpicxx=mpicxx, mpifc=mpifc,
                    esmf_comm=esmf_comm, esmf_netcdf=esmf_netcdf,
                    esmf_compilers=esmf_compilers, netcdf_paths=netcdf_paths,
                    mpas_netcdf_paths=mpas_netcdf_paths, env_vars=env_vars)

    return sys_info


def build_system_libraries(source_path, scorpio, esmf, scorpio_path, esmf_path,
                           sys_info):

    if esmf == 'None':
        esmf_branch = 'None'
    else:
        esmf_branch = 'ESMF_{}'.format(esmf.replace('.', '_'))

    script_filename = 'build.bash'

    with open('{}/deploy/build.template'.format(source_path), 'r') as f:
        template = Template(f.read())

    script = template.render(
        sys_info=sys_info, modules='\n'.join(sys_info['modules']),
        scorpio=scorpio, scorpio_path=scorpio_path,  esmf_path=esmf_path,
        esmf_branch=esmf_branch)
    print('Writing {}'.format(script_filename))
    with open(script_filename, 'w') as handle:
        handle.write(script)

    command = '/bin/bash build.bash'
    check_call(command)


def write_load_compass(source_path, conda_base, script_filename, compass_env,
                       machine, sys_info):

    with open('{}/deploy/load_compass.template'.format(source_path), 'r') as f:
        template = Template(f.read())

    script = template.render(conda_base=conda_base, compass_env=compass_env,
                             modules='\n'.join(sys_info['modules']),
                             env_vars='\n'.join(sys_info['env_vars']),
                             netcdf_paths=sys_info['mpas_netcdf_paths'],
                             script_filename=script_filename, machine=machine)

    print('Writing {}'.format(script_filename))
    with open(script_filename, 'w') as handle:
        handle.write(script)


def check_env(script_filename, env_name, is_test):
    print("Checking the environment {}".format(env_name))

    activate = 'source {}'.format(script_filename)

    imports = ['geometric_features', 'mpas_tools', 'jigsawpy', 'compass']
    commands = [['gpmetis', '--help'],
                ['ffmpeg', '--help']]
    if not is_test:
        commands.extend(
            [['compass', 'list'],
             ['compass', 'setup', '--help'],
             ['compass', 'suite', '--help'],
             ['compass', 'clean', '--help']])
    else:
        commands.extend(
            [['python', '-m', 'compass', 'list'],
             ['python', '-m', 'compass', 'setup', '--help'],
             ['python', '-m', 'compass', 'suite', '--help'],
             ['python', '-m', 'compass', 'clean', '--help']])

    for import_name in imports:
        command = '{}; python -c "import {}"'.format(activate, import_name)
        test_command(command, os.environ, import_name)

    for command in commands:
        package = command[0]
        command = '{}; {}'.format(activate, ' '.join(command))
        test_command(command, os.environ, package)


def test_command(command, env, package):
    try:
        check_call(command, env=env)
    except subprocess.CalledProcessError as e:
        print('  {} failed'.format(package))
        raise e
    print('  {} passes'.format(package))


def check_call(commands, env=None):
    print('runnning: {}'.format(commands))
    subprocess.run(commands, env=env, executable='/bin/bash', shell=True,
                   check=True)


def main():
    parser = argparse.ArgumentParser(
        description='Deploy a compass conda environment')
    parser.add_argument("-m", "--machine", dest="machine",
                        help="The name of the machine for loading machine-"
                             "related config options")
    parser.add_argument("-f", "--config_file", dest="config_file",
                        help="Config file to override deployment config "
                             "options")
    parser.add_argument("--test", dest="test", action='store_true',
                        help="Deploy a test environment")
    parser.add_argument("--no-test", dest="test", action='store_false',
                        help="Deploy a production environment")
    parser.add_argument("--build", dest="build", action='store_true',
                        help="Build the test package")
    parser.add_argument("--no-build", dest="build", action='store_false',
                        help="Don't build the test package")
    parser.add_argument("--recreate", dest="recreate", action='store_true',
                        help="Recreate the environment if it exists")
    parser.add_argument("--no-recreate", dest="recreate", action='store_false',
                        help="Don't recreate the environment if it exists")
    parser.add_argument("--no-clean", dest="clean", action='store_false',
                        default=True,
                        help="Don't delete existing package builds if building"
                             " a new package")
    parser.add_argument("-p", "--python", dest="python", type=str,
                        help="The python version to deploy")
    parser.add_argument("-i", "--mpi", dest="mpi", type=str,
                        help="The MPI library (nompi, mpich, openmpi or a "
                             "system flavor) to deploy")
    parser.add_argument("-c", "--compiler", dest="compiler", type=str,
                        help="The name of the compiler")
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
    machine = args.machine
    if machine is not None:
        machine_config = os.path.join(here, '..', 'compass', 'machines',
                                      '{}.cfg'.format(machine))
        config.read(machine_config)

    if args.config_file is not None:
        config.read(args.config_file)

    if args.group is not None:
        group = args.group
    else:
        group = config.get('deploy', 'group')

    if args.python is not None:
        python = args.python
    else:
        python = config.get('deploy', 'python')

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

    if args.compiler is not None:
        compiler = args.compiler
    elif config.has_option('deploy', 'compiler'):
        compiler = config.get('deploy', 'compiler')
    else:
        compiler = None

    if args.mpi is not None:
        mpi = args.mpi
    else:
        mpi = config.get('deploy', 'mpi_{}'.format(compiler))

    esmf = config.get('deploy', 'esmf')
    scorpio = config.get('deploy', 'scorpio')

    if compiler is not None:
        suffix = '_{}_{}'.format(compiler, mpi)
    else:
        suffix = ''

    source_path = os.getcwd()
    if machine is not None:
        build_dir = 'deploy/build_{}{}'.format(machine, suffix)
        try:
            shutil.rmtree(build_dir)
        except OSError:
            pass
        try:
            os.makedirs(build_dir)
        except FileExistsError:
            pass

        os.chdir(build_dir)

    if compiler is not None:
        sys_info = get_e3sm_compiler_and_mpi(machine, compiler, mpi)
        conda_mpi = 'nompi'
        system_libs = config.get('deploy', 'system_libs')
        compiler_path = os.path.join(system_libs, 'compass_{}'.format(version),
                                     compiler, mpi)
        scorpio_path = os.path.join(compiler_path,
                                    'scorpio_{}'.format(scorpio))
        esmf_path = os.path.join(compiler_path, 'esmf_{}'.format(esmf))
        if esmf != 'None':
            sys_info['env_vars'].append('export PATH="{}:$PATH"'.format(
                os.path.join(esmf_path, 'bin')))

        if scorpio != 'None':
            sys_info['env_vars'].append('export PIO={}'.format(scorpio_path))
    else:
        sys_info = dict(modules=[], env_vars=[], mpas_netcdf_paths='')
        conda_mpi = mpi
        system_libs = None
        scorpio_path = None
        esmf_path = None

    base_path = config.get('paths', 'compass_envs')
    activ_path = os.path.abspath(os.path.join(base_path, '..'))

    if not os.path.exists(base_path):
        miniconda = 'Miniconda3-latest-Linux-x86_64.sh'
        url = 'https://repo.continuum.io/miniconda/{}'.format(miniconda)
        r = requests.get(url)
        with open(miniconda, 'wb') as outfile:
            outfile.write(r.content)

        command = '/bin/bash {} -b -p {}'.format(miniconda, base_path)
        check_call(command)
        os.remove(miniconda)

    activate = 'source {}/etc/profile.d/conda.sh; conda activate'.format(
        base_path)

    print('Doing initial setup')
    commands = '{}; ' \
               'conda config --add channels conda-forge; ' \
               'conda config --set channel_priority strict; ' \
               'conda install -y conda-build; ' \
               'conda update -y --all'.format(activate)

    check_call(commands)
    print('done')

    if is_test and build:
        if args.clean:
            commands = 'rm -rf {}/conda-bld'.format(base_path)
            check_call(commands)
        print('Building the conda package')
        commands = '{}; ' \
                   'conda build -m {}/ci/mpi_{}.yaml {}/recipe'.format(
                       activate, source_path, conda_mpi, source_path)

        check_call(commands)

    if conda_mpi == 'nompi':
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
        env_name = 'test_compass_{}'.format(version)
    else:
        env_name = 'compass_{}'.format(version)
    env_path = os.path.join(base_path, 'envs', env_name)
    if not os.path.exists(env_path) or force_recreate:
        print('creating {}'.format(env_name))
        commands = '{}; conda create -y -n {} {} {}'.format(
            activate, env_name, channels, packages)
        check_call(commands)

        if compiler is not None and esmf != 'None':
            # remove conda-forge esmf because we will use the system build
            commands = '{}; conda remove -y --force -n {} esmf'.format(
                activate, env_name)
            check_call(commands)

        if is_test:
            # remove compass itself because this is a development environment
            commands = '{}; conda remove -y --force -n {} compass'.format(
                activate, env_name)
            check_call(commands)

    else:
        print('{} already exists'.format(env_name))

    if compiler is not None:
        build_system_libraries(source_path, scorpio, esmf, scorpio_path,
                               esmf_path, sys_info)

    try:
        os.makedirs(activ_path)
    except FileExistsError:
        pass

    if is_test:
        prefix = 'test'
    else:
        prefix = 'load'

    script_filename = '{}/{}_compass{}{}.sh'.format(
        activ_path, prefix, version, suffix)
    write_load_compass(source_path, base_path, script_filename, env_name,
                       machine, sys_info)

    os.chdir(source_path)
    check_env(script_filename, env_name, is_test)

    commands = '{}; conda clean -y -p -t'.format(activate)
    check_call(commands)

    print('changing permissions on activation scripts')
    activation_files = glob.glob('{}/*_compass*.sh'.format(
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
    for base in [base_path, system_libs]:
        for root, dirs, files in os.walk(base):
            files_and_dirs.extend(dirs)
            files_and_dirs.extend(files)

    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(),
               ' ', progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets,
                                  maxval=len(files_and_dirs)).start()
    progress = 0
    for base in [base_path, system_libs]:
        for root, dirs, files in os.walk(base):
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
