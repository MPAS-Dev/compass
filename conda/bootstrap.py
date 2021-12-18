#!/usr/bin/env python

from __future__ import print_function

import os
import re
import platform
import subprocess
import glob
import stat
import grp
import shutil
import progressbar
from jinja2 import Template
from importlib.resources import path
from configparser import ConfigParser

from mache import MachineInfo, discover_machine
from mache.spack import make_spack_env, get_modules_env_vars_and_mpi_compilers
from shared import parse_args, get_conda_base, check_call, install_miniconda


def get_config(config_file, machine):
    # we can't load compass so we find the config files
    here = os.path.abspath(os.path.dirname(__file__))
    default_config = os.path.join(here, 'default.cfg')
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


def get_version():
    # we can't import compass because we probably don't have the necessary
    # dependencies, so we get the version by parsing (same approach used in
    # the root setup.py)
    here = os.path.abspath(os.path.dirname(__file__))
    version_path = os.path.join(here, '..', 'compass', 'version.py')
    with open(version_path) as f:
        main_ns = {}
        exec(f.read(), main_ns)
        version = main_ns['__version__']

    return version


def get_env_setup(args, config, machine, env_type, source_path, conda_base):

    if args.python is not None:
        python = args.python
    else:
        python = config.get('deploy', 'python')

    if args.recreate is not None:
        recreate = args.recreate
    else:
        recreate = config.getboolean('deploy', 'recreate')

    if args.compiler is not None:
        compiler = args.compiler
    elif config.has_option('deploy', 'compiler'):
        compiler = config.get('deploy', 'compiler')
    else:
        compiler = None

    if args.mpi is not None:
        mpi = args.mpi
    elif compiler is None:
        mpi = 'nompi'
    else:
        mpi = config.get('deploy', 'mpi_{}'.format(compiler))

    if machine is not None:
        conda_mpi = 'nompi'
    else:
        conda_mpi = mpi

    if machine is not None:
        activ_suffix = '_{}_{}_{}'.format(machine, compiler, mpi)
        env_suffix = ''
    elif conda_mpi != 'nompi':
        activ_suffix = '_{}'.format(mpi)
        env_suffix = activ_suffix
    else:
        activ_suffix = ''
        env_suffix = ''

    if env_type == 'dev':
        activ_path = source_path
    else:
        activ_path = os.path.abspath(os.path.join(conda_base, '..'))

    check_unsupported(machine, compiler, mpi, source_path)

    return python, recreate, compiler, mpi, conda_mpi,  activ_suffix, \
        env_suffix, activ_path


def build_env(env_type, recreate, machine, compiler, mpi, conda_mpi, version,
              python, source_path, template_path, conda_base, activ_suffix,
              env_name, env_suffix, activate_base, use_local,
              local_conda_build):

    if env_type != 'dev':
        install_miniconda(conda_base, activate_base)

    if compiler is not None or env_type == 'dev':
        build_dir = f'conda/build{activ_suffix}'

        try:
            shutil.rmtree(build_dir)
        except OSError:
            pass
        try:
            os.makedirs(build_dir)
        except FileExistsError:
            pass

        os.chdir(build_dir)

    if env_type == 'dev':
        spack_env = f'dev_compass_{version}{env_suffix}'
    elif env_type == 'test_release':
        spack_env = f'test_compass_{version}{env_suffix}'
    else:
        spack_env = f'compass_{version}{env_suffix}'

    if env_name is None or env_type != 'dev':
        env_name = spack_env

    # add the compiler and MPI library to the spack env name
    spack_env = '{}_{}_{}'.format(spack_env, compiler, mpi)
    # spack doesn't like dots
    spack_env = spack_env.replace('.', '_')

    env_path = os.path.join(conda_base, 'envs', env_name)

    if conda_mpi == 'nompi':
        mpi_prefix = 'nompi'
    else:
        mpi_prefix = f'mpi_{mpi}'

    channels = ['-c conda-forge', '-c defaults']
    if use_local:
        channels = ['--use-local'] + channels
    if local_conda_build is not None:
        channels = ['-c', local_conda_build] + channels
    if env_type == 'test_release':
        # for a test release, we will be the compass package from the dev label
        channels = channels + ['-c e3sm/label/compass_dev']
    if machine is None or env_type == 'release':
        # we need libpnetcdf and scorpio (and maybe compass itself) from the
        # e3sm channel, compass label
        channels = channels + ['-c e3sm/label/compass']

    channels = f'--override-channels {" ".join(channels)}'
    packages = f'python={python}'

    base_activation_script = os.path.abspath(
        f'{conda_base}/etc/profile.d/conda.sh')

    activate_env = \
        f'source {base_activation_script}; conda activate {env_name}'

    with open(f'{template_path}/spec-file.template', 'r') as f:
        template = Template(f.read())

    if env_type == 'dev':
        spec_file = template.render(mpi=conda_mpi, mpi_prefix=mpi_prefix)

        spec_filename = f'spec-file-{conda_mpi}.txt'
        with open(spec_filename, 'w') as handle:
            handle.write(spec_file)
    else:
        spec_filename = None

    if not os.path.exists(env_path) or recreate:
        print(f'creating {env_name}')
        if env_type == 'dev':
            # install dev dependencies and compass itself
            commands = \
                f'{activate_base}; ' \
                f'mamba create -y -n {env_name} {channels} ' \
                f'--file {spec_filename} {packages}'
            check_call(commands)

            commands = \
                f'{activate_env}; ' \
                f'cd {source_path}; ' \
                f'python -m pip install -e .'
            check_call(commands)

        else:
            # conda packages don't like dashes
            version_conda = version.replace('-', '')
            packages = f'{packages} "compass={version_conda}={mpi_prefix}_*"'
            commands = f'{activate_base}; ' \
                       f'mamba create -y -n {env_name} {channels} {packages}'
            check_call(commands)
    else:
        if env_type == 'dev':
            print(f'updating {env_name}')
            # install dev dependencies and compass itself
            commands = \
                f'{activate_base}; ' \
                f'mamba install -y -n {env_name} {channels} ' \
                f'--file {spec_filename} {packages}'
            check_call(commands)

            commands = \
                f'{activate_env}; ' \
                f'cd {source_path}; ' \
                f'python -m pip install -e .'
            check_call(commands)
        else:
            print(f'{env_name} already exists')

    return env_path, env_name, activate_env, spack_env


def get_sys_info(machine, compiler, mpilib, mpicc, mpicxx, mpifc,
                 mod_env_commands):

    if machine is None:
        machine = 'None'

    # convert env vars from mache to a list

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
        mod_env_commands = f'{mod_env_commands}\n' \
                           f'export I_MPI_CC=icc\n'  \
                           f'export I_MPI_CXX=icpc\n'  \
                           f'export I_MPI_F77=ifort\n'  \
                           f'export I_MPI_F90=ifort'

    if platform.system() == 'Linux' and machine == 'None':
        mod_env_commands = f'{mod_env_commands}\n' \
                           f'export MPAS_EXTERNAL_LIBS="-lgomp"'

    if mpilib == 'mvapich':
        esmf_comm = 'mvapich2'
        mod_env_commands = f'{mod_env_commands}\n' \
                           f'export MV2_ENABLE_AFFINITY=0' \
                           f'export MV2_SHOW_CPU_BINDING=1'
    elif mpilib == 'mpich':
        esmf_comm = 'mpich3'
    elif mpilib == 'impi':
        esmf_comm = 'intelmpi'
    else:
        esmf_comm = mpilib

    if machine == 'grizzly' or machine == 'badger':
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

    sys_info = dict(mpicc=mpicc, mpicxx=mpicxx,
                    mpifc=mpifc, esmf_comm=esmf_comm, esmf_netcdf=esmf_netcdf,
                    esmf_compilers=esmf_compilers, netcdf_paths=netcdf_paths,
                    mpas_netcdf_paths=mpas_netcdf_paths)

    return sys_info, mod_env_commands


def build_spack_env(config, update_spack, machine, compiler, mpi, env_path,
                    env_name, activate_env, spack_env, mod_env_commands):

    if machine is not None:
        esmf = config.get('deploy', 'esmf')
        scorpio = config.get('deploy', 'scorpio')
    else:
        # stick with the conda-forge ESMF and e3sm/label/compass SCORPIO
        esmf = 'None'
        scorpio = 'None'

    if esmf != 'None':
        # remove conda-forge esmf because we will use the system build
        commands = '{}; conda remove -y --force -n {} esmf'.format(
            activate_env, env_name)
        try:
            check_call(commands)
        except subprocess.CalledProcessError:
            # it could be that esmf was already removed
            pass

    if machine is not None:
        spack_base = config.get('deploy', 'spack')
        scorpio_path = os.path.join(spack_base, 'var/spack/environments/',
                                    spack_env, '.spack-env/view')
    else:
        spack_base = None
        scorpio_path = env_path

    mod_env_commands = f'{mod_env_commands}\n' \
                       f'export PIO={scorpio_path}'

    specs = list()

    if esmf != 'None':
        specs.append(f'esmf@{esmf}+mpi+netcdf~pio+pnetcdf')
    if scorpio != 'None':
        specs.append(f'scorpio@{scorpio}+pnetcdf~timing+internal-timing~tools+malloc')

    if update_spack:
        make_spack_env(spack_path=spack_base, env_name=spack_env,
                       spack_specs=specs, compiler=compiler, mpi=mpi,
                       machine=machine)

        # remove ESMC/ESMF include files that interfere with MPAS time keeping
        include_path = os.path.join(scorpio_path, 'include')
        for prefix in ['ESMC', 'esmf']:
            files = glob.glob(os.path.join(include_path, f'{prefix}*'))
            for filename in files:
                os.remove(filename)

    return spack_base, mod_env_commands


def write_load_compass(template_path, activ_path, conda_base, env_type,
                       activ_suffix, prefix, env_name, spack_base, spack_env,
                       machine, sys_info, mod_env_commands, env_only):

    try:
        os.makedirs(activ_path)
    except FileExistsError:
        pass

    if prefix.endswith(activ_suffix):
        # avoid a redundant activation script name if the suffix is already
        # part of the environment name
        script_filename = '{}/{}.sh'.format(activ_path, prefix)
    else:
        script_filename = '{}/{}{}.sh'.format(activ_path, prefix, activ_suffix)

    if not env_only:
        mod_env_commands = f'{mod_env_commands}\n' \
                           f'export USE_PIO2=true'
    mod_env_commands = f'{mod_env_commands}\n' \
                       f'export HDF5_USE_FILE_LOCKING=FALSE\n' \
                       f'export LOAD_COMPASS_ENV={script_filename}'
    if machine is not None:
        mod_env_commands = f'{mod_env_commands}\n' \
                           f'export COMPASS_MACHINE={machine}'

    filename = '{}/load_compass.template'.format(template_path)
    with open(filename, 'r') as f:
        template = Template(f.read())

    if not env_only and spack_base is not None:
        spack = f'source {spack_base}/share/spack/setup-env.sh\n' \
                f'spack env activate {spack_env}'
    else:
        spack = ''

    if env_type == 'dev':
        update_compass = \
            'if [[ -f "./setup.py" && -d "compass" ]]; then\n' \
            '   # safe to assume we\'re in the compass repo\n' \
            '   # update the compass installation to point here\n' \
            '   python -m pip install -e .\n' \
            'fi'
    else:
        update_compass = ''

    script = template.render(conda_base=conda_base, compass_env=env_name,
                             mod_env_commands=mod_env_commands,
                             spack=spack,
                             netcdf_paths=sys_info['mpas_netcdf_paths'],
                             update_compass=update_compass)

    # strip out redundant blank lines
    lines = list()
    prev_line = ''
    for line in script.split('\n'):
        line = line.strip()
        if line != '' or prev_line != '':
            lines.append(line)
        prev_line = line

    lines.append('')

    script = '\n'.join(lines)

    print('Writing {}'.format(script_filename))
    with open(script_filename, 'w') as handle:
        handle.write(script)

    return script_filename


def check_env(script_filename, env_name):
    print("Checking the environment {}".format(env_name))

    activate = 'source {}'.format(script_filename)

    imports = ['geometric_features', 'mpas_tools', 'jigsawpy', 'compass']
    commands = [['gpmetis', '--help'],
                ['ffmpeg', '--help'],
                ['compass', 'list'],
                ['compass', 'setup', '--help'],
                ['compass', 'suite', '--help'],
                ['compass', 'clean', '--help']]

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


def update_permissions(config, env_type, activ_path, conda_base, spack_base):

    directories = []
    if env_type != 'dev':
        directories.append(conda_base)
    if spack_base is not None:
        # even if this is not a release, we need to update permissions on
        # shared system libraries
        directories.append(spack_base)

    group = config.get('e3sm_unified', 'group')

    new_uid = os.getuid()
    new_gid = grp.getgrnam(group).gr_gid

    print('changing permissions on activation scripts')

    read_perm = (stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP |
                 stat.S_IWGRP | stat.S_IROTH)
    exec_perm = (stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR |
                 stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP |
                 stat.S_IROTH | stat.S_IXOTH)

    mask = stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO

    if env_type != 'dev':

        activation_files = glob.glob('{}/*_compass*.sh'.format(
            activ_path))
        for file_name in activation_files:
            os.chmod(file_name, read_perm)
            os.chown(file_name, new_uid, new_gid)

    print('changing permissions on environments')

    # first the base directories that don't seem to be included in
    # os.walk()
    for directory in directories:
        try:
            dir_stat = os.stat(directory)
        except OSError:
            continue

        perm = dir_stat.st_mode & mask

        if dir_stat.st_uid != new_uid:
            # current user doesn't own this dir so let's move on
            continue

        if perm == exec_perm and dir_stat.st_gid == new_gid:
            continue

        try:
            os.chown(directory, new_uid, new_gid)
            os.chmod(directory, exec_perm)
        except OSError:
            continue

    files_and_dirs = []
    for base in directories:
        for root, dirs, files in os.walk(base):
            files_and_dirs.extend(dirs)
            files_and_dirs.extend(files)

    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(),
               ' ', progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets,
                                  maxval=len(files_and_dirs)).start()
    progress = 0
    for base in directories:
        for root, dirs, files in os.walk(base):
            for directory in dirs:
                progress += 1
                try:
                    bar.update(progress)
                except ValueError:
                    pass

                directory = os.path.join(root, directory)

                try:
                    dir_stat = os.stat(directory)
                except OSError:
                    continue

                if dir_stat.st_uid != new_uid:
                    # current user doesn't own this dir so let's move on
                    continue

                perm = dir_stat.st_mode & mask

                if perm == exec_perm and dir_stat.st_gid == new_gid:
                    continue

                try:
                    os.chown(directory, new_uid, new_gid)
                    os.chmod(directory, exec_perm)
                except OSError:
                    continue

            for file_name in files:
                progress += 1
                try:
                    bar.update(progress)
                except ValueError:
                    pass
                file_name = os.path.join(root, file_name)
                try:
                    file_stat = os.stat(file_name)
                except OSError:
                    continue

                if file_stat.st_uid != new_uid:
                    # current user doesn't own this file so let's move on
                    continue

                perm = file_stat.st_mode & mask

                if perm & stat.S_IXUSR:
                    # executable, so make sure others can execute it
                    new_perm = exec_perm
                else:
                    new_perm = read_perm

                if perm == new_perm and file_stat.st_gid == new_gid:
                    continue

                try:
                    os.chown(file_name, new_uid, new_gid)
                    os.chmod(file_name, new_perm)
                except OSError:
                    continue

    bar.finish()
    print('  done.')


def check_unsupported(machine, compiler, mpi, source_path):
    with open(os.path.join(source_path, 'conda', 'unsupported.txt'), 'r') as f:
        content = f.readlines()
    content = [line.strip() for line in content if not line.startswith('#')]
    for line in content:
        if line.strip() == '':
            continue
        unsupported = [part.strip() for part in line.split(',')]
        if len(unsupported) != 3:
            raise ValueError('Bad line in "unsupported.txt" {}'.format(line))
        if machine == unsupported[0] and compiler == unsupported[1] and \
                mpi == unsupported[2]:
            raise ValueError('{} with {} is not supported on {}'.format(
                compiler, mpi, machine))


def main():
    args = parse_args(bootstrap=True)

    source_path = os.getcwd()
    template_path = '{}/conda/compass_env'.format(source_path)

    version = get_version()

    machine = None
    machine_info = None
    if not args.env_only:
        if args.machine is None:
            machine = discover_machine()
        else:
            machine = args.machine
        if machine is not None:
            machine_info = MachineInfo(machine=machine)

    config = get_config(args.config_file, machine)

    env_type = config.get('deploy', 'env_type')
    if env_type not in ['dev', 'test_release', 'release']:
        raise ValueError(f'Unexpected env_type: {env_type}')
    shared = (env_type != 'dev')
    conda_base = get_conda_base(args.conda_base, config, shared=shared)

    base_activation_script = os.path.abspath(
        '{}/etc/profile.d/conda.sh'.format(conda_base))

    activate_base = 'source {}; conda activate'.format(base_activation_script)

    python, recreate, compiler, mpi, conda_mpi, activ_suffix, env_suffix, \
        activ_path = get_env_setup(args, config, machine, env_type,
                                   source_path, conda_base)

    if machine is None and not args.env_only and args.mpi is None:
        raise ValueError('Your machine wasn\'t recognized by compass but you '
                         'didn\'t specify the MPI version. Please provide '
                         'either the --mpi or --env_only flag.')

    if machine is None:
        if args.env_only:
            compiler = None
        elif platform.system() == 'Linux':
            compiler = 'gnu'
        elif platform.system() == 'Darwin':
            compiler = 'clang'
        else:
            compiler = 'gnu'

    if machine_info is not None:
        mpicc, mpicxx, mpifc, mod_env_commands = \
            get_modules_env_vars_and_mpi_compilers(
                machine, compiler, mpi, shell='sh',
                include_e3sm_hdf5_netcdf=True)
    else:
        # using conda-forge compilers
        mpicc = 'mpicc'
        mpicxx = 'mpicxx'
        mpifc = 'mpifort'
        mod_env_commands = ''

    env_path, env_name, activate_env, spack_env = build_env(
        env_type, recreate, machine, compiler, mpi, conda_mpi, version, python,
        source_path, template_path, conda_base, activ_suffix, args.env_name,
        env_suffix, activate_base, args.use_local, args.local_conda_build)

    if compiler is not None:
        sys_info, mod_env_commands = get_sys_info(
            machine, compiler, mpi, mpicc, mpicxx, mpifc, mod_env_commands)
        spack_base = build_spack_env(
            config, args.update_spack, machine, compiler, mpi, env_path,
            env_name, activate_env, spack_env, sys_info)
    else:
        sys_info = dict(mpas_netcdf_paths='')
        spack_base = None
        mod_env_commands = ''

    if env_type == 'dev':
        if args.env_name is not None:
            prefix = 'load_{}'.format(args.env_name)
        else:
            prefix = 'load_dev_compass_{}'.format(version)
    elif env_type == 'test_release':
        prefix = 'test_compass_{}'.format(version)
    else:
        prefix = 'load_compass_{}'.format(version)

    script_filename = write_load_compass(
        template_path, activ_path, conda_base, env_type, activ_suffix, prefix,
        env_name, spack_base, spack_env, machine, sys_info, mod_env_commands,
        args.env_only)

    if args.check:
        check_env(script_filename, env_name)

    if env_type == 'release':
        # make a symlink to the activation script
        link = os.path.join(activ_path,
                            f'load_latest_compass_{compiler}_{mpi}.sh')
        check_call(f'ln -sfn {script_filename} {link}')

        default_compiler = config.get('deploy', 'compiler')
        default_mpi = config.get('deploy', 'mpi_{}'.format(default_compiler))
        if compiler == default_compiler and mpi == default_mpi:
            # make a default symlink to the activation script
            link = os.path.join(activ_path, 'load_latest_compass.sh')
            check_call(f'ln -sfn {script_filename} {link}')

    commands = '{}; conda clean -y -p -t'.format(activate_base)
    check_call(commands)

    if machine is not None:
        update_permissions(config, env_type, activ_path, conda_base,
                           spack_base)


if __name__ == '__main__':
    main()
