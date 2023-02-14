#!/usr/bin/env python

from __future__ import print_function

import os
import platform
import subprocess
import glob
import stat
import grp
import shutil
import progressbar
from jinja2 import Template
import importlib.resources
from configparser import ConfigParser

from mache import discover_machine, MachineInfo
from mache.spack import make_spack_env, get_spack_script
from mache.version import __version__ as mache_version
from shared import parse_args, get_conda_base, get_spack_base, check_call, \
    install_miniconda, get_logger


def get_config(config_file, machine):
    # we can't load compass so we find the config files
    here = os.path.abspath(os.path.dirname(__file__))
    default_config = os.path.join(here, 'default.cfg')
    config = ConfigParser()
    config.read(default_config)

    if machine is not None:
        if not machine.startswith('conda'):
            machine_config = \
                importlib.resources.files('mache.machines') / f'{machine}.cfg'
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


def get_compilers_mpis(config, machine, compilers, mpis, source_path):

    unsupported = parse_unsupported(machine, source_path)
    if machine is None:
        all_compilers = None
        all_mpis = None
    elif machine == 'conda-linux':
        all_compilers = ['gfortran']
        all_mpis = ['mpich', 'openmpi']
    elif machine == 'conda-osx':
        all_compilers = ['clang']
        all_mpis = ['mpich', 'openmpi']
    else:
        machine_info = MachineInfo(machine)
        all_compilers = machine_info.compilers
        all_mpis = machine_info.mpilibs

    if config.has_option('deploy', 'compiler'):
        default_compiler = config.get('deploy', 'compiler')
    else:
        default_compiler = None

    error_on_unsupported = True

    if compilers is not None and compilers[0] == 'all':
        error_on_unsupported = False
        if mpis is not None and mpis[0] == 'all':
            # make a matrix of compilers and mpis
            compilers = list()
            mpis = list()
            for compiler in all_compilers:
                for mpi in all_mpis:
                    compilers.append(compiler)
                    mpis.append(mpi)
        else:
            compilers = all_compilers
            if mpis is not None:
                if len(mpis) > 1:
                    raise ValueError(f'"--compiler all" can only be combined '
                                     f'with "--mpi all" or a single MPI '
                                     f'library, \n'
                                     f'but got: {mpis}')
                mpi = mpis[0]
                mpis = [mpi for _ in compilers]

    elif mpis is not None and mpis[0] == 'all':
        error_on_unsupported = False
        mpis = all_mpis
        if compilers is None:
            compiler = default_compiler
        else:
            if len(compilers) > 1:
                raise ValueError(f'"--mpis all" can only be combined with '
                                 f'"--compiler all" or a single compiler, \n'
                                 f'but got: {compilers}')
            compiler = compilers[0]
        # The compiler is all the same
        compilers = [compiler for _ in mpis]

    if compilers is None:
        if config.has_option('deploy', 'compiler'):
            compilers = [config.get('deploy', 'compiler')]
        else:
            compilers = [None]

    if mpis is None:
        mpis = list()
        for compiler in compilers:
            if compiler is None:
                mpi = 'nompi'
            else:
                mpi = config.get('deploy', 'mpi_{}'.format(compiler))
            mpis.append(mpi)

    supported_compilers = list()
    supported_mpis = list()
    for compiler, mpi in zip(compilers, mpis):
        if (compiler, mpi) in unsupported:
            if error_on_unsupported:
                raise ValueError(f'{compiler} with {mpi} is not supported on '
                                 f'{machine}')
        else:
            supported_compilers.append(compiler)
            supported_mpis.append(mpi)

    return supported_compilers, supported_mpis


def get_env_setup(args, config, machine, compiler, mpi, env_type, source_path,
                  conda_base, env_name, compass_version, logger):

    if args.python is not None:
        python = args.python
    else:
        python = config.get('deploy', 'python')

    if args.recreate is not None:
        recreate = args.recreate
    else:
        recreate = config.getboolean('deploy', 'recreate')

    if machine is None:
        conda_mpi = 'nompi'
        activ_suffix = ''
        env_suffix = ''
    elif not machine.startswith('conda'):
        conda_mpi = 'nompi'
        activ_suffix = '_{}_{}_{}'.format(machine, compiler, mpi)
        env_suffix = ''
    else:
        activ_suffix = '_{}'.format(mpi)
        env_suffix = activ_suffix
        conda_mpi = mpi

    lib_suffix = ''
    if args.with_albany:
        lib_suffix = f'{lib_suffix}_albany'
    else:
        config.set('deploy', 'albany', 'None')

    if args.with_netlib_lapack:
        lib_suffix = f'{lib_suffix}_netlib_lapack'
    else:
        config.set('deploy', 'lapack', 'None')

    if args.with_petsc:
        lib_suffix = f'{lib_suffix}_petsc'
        logger.info('Turning off OpenMP because it doesn\'t work well '
                    'with  PETSc')
        args.without_openmp = True
    else:
        config.set('deploy', 'petsc', 'None')

    activ_suffix = f'{activ_suffix}{lib_suffix}'

    if env_type == 'dev':
        activ_path = source_path
    else:
        activ_path = os.path.abspath(os.path.join(conda_base, '..'))

    if args.with_albany:
        check_supported('albany', machine, compiler, mpi, source_path)

    if args.with_petsc:
        check_supported('petsc', machine, compiler, mpi, source_path)

    if env_type == 'dev':
        spack_env = f'dev_compass_{compass_version}{env_suffix}'
    elif env_type == 'test_release':
        spack_env = f'test_compass_{compass_version}{env_suffix}'
    else:
        spack_env = f'compass_{compass_version}{env_suffix}'

    if env_name is None or env_type != 'dev':
        env_name = spack_env

    # add the compiler and MPI library to the spack env name
    spack_env = f'{spack_env}_{compiler}_{mpi}{lib_suffix}'
    # spack doesn't like dots
    spack_env = spack_env.replace('.', '_')

    env_path = os.path.join(conda_base, 'envs', env_name)

    source_activation_scripts = \
        f'source {conda_base}/etc/profile.d/conda.sh; ' \
        f'source {conda_base}/etc/profile.d/mamba.sh'

    activate_env = f'{source_activation_scripts}; conda activate {env_name}'

    return python, recreate, conda_mpi,  activ_suffix, env_suffix, \
        activ_path, env_path, env_name, activate_env, spack_env


def build_conda_env(env_type, recreate, machine, mpi, conda_mpi, version,
                    python, source_path, conda_template_path, conda_base,
                    env_name, env_path, activate_base, use_local,
                    local_conda_build, logger):

    if env_type != 'dev':
        install_miniconda(conda_base, activate_base, logger)

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
    channels = channels + ['-c e3sm/label/compass']

    channels = f'--override-channels {" ".join(channels)}'
    packages = f'python={python}'

    base_activation_script = os.path.abspath(
        f'{conda_base}/etc/profile.d/conda.sh')

    activate_env = \
        f'source {base_activation_script}; conda activate {env_name}'

    with open(f'{conda_template_path}/spec-file.template', 'r') as f:
        template = Template(f.read())

    if env_type == 'dev':
        supports_otps = platform.system() == 'Linux'
        if platform.system() == 'Linux':
            conda_openmp = 'libgomp'
        elif platform.system() == 'Darwin':
            conda_openmp = 'llvm-openmp'
        else:
            conda_openmp = ''
        spec_file = template.render(supports_otps=supports_otps,
                                    mpi=conda_mpi, openmp=conda_openmp,
                                    mpi_prefix=mpi_prefix)

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
            check_call(commands, logger=logger)

            commands = \
                f'{activate_env}; ' \
                f'cd {source_path}; ' \
                f'python -m pip install -e .'
            check_call(commands, logger=logger)

        else:
            # conda packages don't like dashes
            version_conda = version.replace('-', '')
            packages = f'{packages} "compass={version_conda}={mpi_prefix}_*"'
            commands = f'{activate_base}; ' \
                       f'mamba create -y -n {env_name} {channels} {packages}'
            check_call(commands, logger=logger)
    else:
        if env_type == 'dev':
            print(f'Updating {env_name}\n')
            # install dev dependencies and compass itself
            commands = \
                f'{activate_base}; ' \
                f'mamba install -y -n {env_name} {channels} ' \
                f'--file {spec_filename} {packages}'
            check_call(commands, logger=logger)

            commands = \
                f'{activate_env}; ' \
                f'cd {source_path}; ' \
                f'python -m pip install -e .'
            check_call(commands, logger=logger)
        else:
            print(f'{env_name} already exists')


def get_env_vars(machine, compiler, mpilib):

    if machine is None:
        machine = 'None'

    # convert env vars from mache to a list
    env_vars = 'export MPAS_EXTERNAL_LIBS=""\n'

    if 'intel' in compiler and machine == 'anvil':
        env_vars = f'{env_vars}' \
                   f'export I_MPI_CC=icc\n' \
                   f'export I_MPI_CXX=icpc\n' \
                   f'export I_MPI_F77=ifort\n' \
                   f'export I_MPI_F90=ifort\n'

    if platform.system() == 'Linux' and machine.startswith('conda'):
        env_vars = \
            f'{env_vars}' \
            f'export MPAS_EXTERNAL_LIBS="${{MPAS_EXTERNAL_LIBS}} -lgomp"\n'

    if mpilib == 'mvapich':
        env_vars = f'{env_vars}' \
                   f'export MV2_ENABLE_AFFINITY=0\n' \
                   f'export MV2_SHOW_CPU_BINDING=1\n'

    if machine.startswith('chicoma') or machine.startswith('pm'):
        env_vars = \
            f'{env_vars}' \
            f'export NETCDF=${{CRAY_NETCDF_HDF5PARALLEL_PREFIX}}\n' \
            f'export NETCDFF=${{CRAY_NETCDF_HDF5PARALLEL_PREFIX}}\n' \
            f'export PNETCDF=${{CRAY_PARALLEL_NETCDF_PREFIX}}\n'
    else:
        env_vars = \
            f'{env_vars}' \
            f'export NETCDF=$(dirname $(dirname $(which nc-config)))\n' \
            f'export NETCDFF=$(dirname $(dirname $(which nf-config)))\n' \
            f'export PNETCDF=$(dirname $(dirname $(which pnetcdf-config)))\n'

    return env_vars


def build_spack_env(config, update_spack, machine, compiler, mpi, spack_env,
                    spack_base, spack_template_path, env_vars, tmpdir, logger):

    albany = config.get('deploy', 'albany')
    esmf = config.get('deploy', 'esmf')
    lapack = config.get('deploy', 'lapack')
    petsc = config.get('deploy', 'petsc')
    scorpio = config.get('deploy', 'scorpio')

    spack_branch_base = f'{spack_base}/spack_for_mache_{mache_version}'

    specs = list()

    e3sm_hdf5_netcdf = config.getboolean('deploy', 'use_e3sm_hdf5_netcdf')
    if not e3sm_hdf5_netcdf:
        hdf5 = config.get('deploy', 'hdf5')
        netcdf_c = config.get('deploy', 'netcdf_c')
        netcdf_fortran = config.get('deploy', 'netcdf_fortran')
        pnetcdf = config.get('deploy', 'pnetcdf')
        specs.extend([
            f'hdf5@{hdf5}+cxx+fortran+hl+mpi+shared',
            f'netcdf-c@{netcdf_c}+mpi~parallel-netcdf',
            f'netcdf-fortran@{netcdf_fortran}',
            f'parallel-netcdf@{pnetcdf}+cxx+fortran'])

    if esmf != 'None':
        specs.append(f'esmf@{esmf}+mpi+netcdf~pio+pnetcdf')
    if lapack != 'None':
        specs.append(f'netlib-lapack@{lapack}')
        include_e3sm_lapack = False
    else:
        include_e3sm_lapack = True
    if petsc != 'None':
        specs.append(f'petsc@{petsc}+mpi+batch')

    if scorpio != 'None':
        specs.append(f'scorpio@{scorpio}+pnetcdf~timing+internal-timing~tools+malloc')

    if albany != 'None':
        specs.append(f'albany@{albany}+mpas+cxx17')

    yaml_template = f'{spack_template_path}/{machine}_{compiler}_{mpi}.yaml'
    if not os.path.exists(yaml_template):
        yaml_template = None
    if update_spack:
        make_spack_env(spack_path=spack_branch_base, env_name=spack_env,
                       spack_specs=specs, compiler=compiler, mpi=mpi,
                       machine=machine,
                       include_e3sm_lapack=include_e3sm_lapack,
                       include_e3sm_hdf5_netcdf=e3sm_hdf5_netcdf,
                       yaml_template=yaml_template, tmpdir=tmpdir)

        # remove ESMC/ESMF include files that interfere with MPAS time keeping
        include_path = f'{spack_branch_base}/var/spack/environments/' \
                       f'{spack_env}/.spack-env/view/include'
        for prefix in ['ESMC', 'esmf']:
            files = glob.glob(os.path.join(include_path, f'{prefix}*'))
            for filename in files:
                os.remove(filename)
        set_ld_library_path(spack_branch_base, spack_env, logger)

    spack_script = get_spack_script(
        spack_path=spack_branch_base, env_name=spack_env, compiler=compiler,
        mpi=mpi, shell='sh', machine=machine,
        include_e3sm_lapack=include_e3sm_lapack,
        include_e3sm_hdf5_netcdf=e3sm_hdf5_netcdf,
        yaml_template=yaml_template)

    spack_view = f'{spack_branch_base}/var/spack/environments/' \
                 f'{spack_env}/.spack-env/view'
    env_vars = f'{env_vars}' \
               f'export PIO={spack_view}\n'
    if albany != 'None':
        albany_flag_filename = f'{spack_view}/export_albany.in'
        if not os.path.exists(albany_flag_filename):
            raise ValueError(f'Missing Albany linking flags in '
                             f'{albany_flag_filename}.\n Maybe your Spack '
                             f'environment may need to be rebuilt with '
                             f'Albany?')
        with open(albany_flag_filename, 'r') as f:
            albany_flags = f.read()
        if platform.system() == 'Darwin':
            stdcxx = '-lc++'
        else:
            stdcxx = '-lstdc++'
        if mpi == 'openmpi' and machine in ['anvil', 'chrysalis']:
            mpicxx = '-lmpi_cxx'
        else:
            mpicxx = ''
        env_vars = \
            f'{env_vars}' \
            f'export {albany_flags}\n' \
            f'export MPAS_EXTERNAL_LIBS="${{MPAS_EXTERNAL_LIBS}} ' \
            f'${{ALBANY_LINK_LIBS}} {stdcxx} {mpicxx}"\n'

    if lapack != 'None':
        env_vars = f'{env_vars}' \
                   f'export LAPACK={spack_view}\n' \
                   f'export USE_LAPACK=true\n'

    if petsc != 'None':
        env_vars = f'{env_vars}' \
                   f'export PETSC={spack_view}\n' \
                   f'export USE_PETSC=true\n'

    return spack_branch_base, spack_script, env_vars


def set_ld_library_path(spack_branch_base, spack_env, logger):
    commands = \
        f'source {spack_branch_base}/share/spack/setup-env.sh; ' \
        f'spack env activate {spack_env}; ' \
        f'spack config add modules:prefix_inspections:lib:[LD_LIBRARY_PATH]; ' \
        f'spack config add modules:prefix_inspections:lib64:[LD_LIBRARY_PATH]'
    check_call(commands, logger=logger)


def write_load_compass(template_path, activ_path, conda_base, env_type,
                       activ_suffix, prefix, env_name, spack_script, machine,
                       env_vars, env_only, source_path, without_openmp):

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
        env_vars = f'{env_vars}\n' \
                   f'export USE_PIO2=true'
    if without_openmp:
        env_vars = f'{env_vars}\n' \
                   f'export OPENMP=false'
    else:
        env_vars = f'{env_vars}\n' \
                   f'export OPENMP=true'

    env_vars = f'{env_vars}\n' \
               f'export HDF5_USE_FILE_LOCKING=FALSE\n' \
               f'export LOAD_COMPASS_ENV={script_filename}'
    if machine is not None and not machine.startswith('conda'):
        env_vars = f'{env_vars}\n' \
                   f'export COMPASS_MACHINE={machine}'

    if env_type == 'dev':
        env_vars = f'{env_vars}\n' \
                   f'export COMPASS_BRANCH={source_path}'

    filename = f'{template_path}/load_compass.template'
    with open(filename, 'r') as f:
        template = Template(f.read())

    if env_type == 'dev':
        update_compass = \
            """
            if [[ -z "${NO_COMPASS_REINSTALL}" && -f "./setup.py" && \\
                  -d "compass" ]]; then
               # safe to assume we're in the compass repo
               # update the compass installation to point here
               mkdir -p conda/logs
               echo Reinstalling compass package in edit mode...
               python -m pip install -e . &> conda/logs/install_compass.log
               echo Done.
               echo
            fi
            """
    else:
        update_compass = ''

    script = template.render(conda_base=conda_base, compass_env=env_name,
                             env_vars=env_vars,
                             spack=spack_script,
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

    print('Writing:\n   {}\n'.format(script_filename))
    with open(script_filename, 'w') as handle:
        handle.write(script)

    return script_filename


def check_env(script_filename, env_name, logger):
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
        test_command(command, os.environ, import_name, logger)

    for command in commands:
        package = command[0]
        command = '{}; {}'.format(activate, ' '.join(command))
        test_command(command, os.environ, package, logger)


def test_command(command, env, package, logger):
    try:
        check_call(command, env=env, logger=logger)
    except subprocess.CalledProcessError as e:
        print('  {} failed'.format(package))
        raise e
    print('  {} passes'.format(package))


def update_permissions(config, env_type, activ_path, directories):

    if not config.has_option('e3sm_unified', 'group'):
        return

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


def parse_unsupported(machine, source_path):
    with open(os.path.join(source_path, 'conda', 'unsupported.txt'), 'r') as f:
        content = f.readlines()
    content = [line.strip() for line in content if not
               line.strip().startswith('#')]
    unsupported = list()
    for line in content:
        if line.strip() == '':
            continue
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != 3:
            raise ValueError(f'Bad line in "unsupported.txt" {line}')
        if parts[0] != machine:
            continue
        compiler = parts[1]
        mpi = parts[2]
        unsupported.append((compiler, mpi))

    return unsupported


def check_supported(library, machine, compiler, mpi, source_path):
    filename = os.path.join(source_path, 'conda', f'{library}_supported.txt')
    with open(filename, 'r') as f:
        content = f.readlines()
    content = [line.strip() for line in content if not
               line.strip().startswith('#')]
    for line in content:
        if line.strip() == '':
            continue
        supported = [part.strip() for part in line.split(',')]
        if len(supported) != 3:
            raise ValueError(f'Bad line in "{library}_supported.txt" {line}')
        if machine == supported[0] and compiler == supported[1] and \
                mpi == supported[2]:
            return

    raise ValueError(f'{compiler} with {mpi} is not supported with {library} '
                     f'on {machine}')


def main():
    args = parse_args(bootstrap=True)

    logger = get_logger(log_filename='conda/logs/bootstrap.log',
                        name=__name__)

    source_path = os.getcwd()
    conda_template_path = f'{source_path}/conda/compass_env'
    spack_template_path = f'{source_path}/conda/spack'

    compass_version = get_version()

    machine = None
    if not args.env_only:
        if args.machine is None:
            machine = discover_machine()
        else:
            machine = args.machine

    e3sm_machine = machine is not None

    if machine is None and not args.env_only:
        if platform.system() == 'Linux':
            machine = 'conda-linux'
        elif platform.system() == 'Darwin':
            machine = 'conda-osx'

    config = get_config(args.config_file, machine)

    env_type = config.get('deploy', 'env_type')
    if env_type not in ['dev', 'test_release', 'release']:
        raise ValueError(f'Unexpected env_type: {env_type}')
    shared = (env_type != 'dev')
    conda_base = get_conda_base(args.conda_base, config, shared=shared,
                                warn=False)
    conda_base = os.path.abspath(conda_base)

    source_activation_scripts = \
        f'source {conda_base}/etc/profile.d/conda.sh; ' \
        f'source {conda_base}/etc/profile.d/mamba.sh'

    activate_base = f'{source_activation_scripts}; conda activate'

    compilers, mpis = get_compilers_mpis(config, machine, args.compilers,
                                         args.mpis, source_path)

    if machine is not None:
        # write out a log file for use by matrix builds
        with open('conda/logs/matrix.log', 'w') as f:
            f.write(f'{machine}\n')
            for compiler, mpi in zip(compilers, mpis):
                f.write(f'{compiler}, {mpi}\n')

    print('Configuring environment(s) for the following compilers and MPI '
          'libraries:')
    for compiler, mpi in zip(compilers, mpis):
        print(f'  {compiler}, {mpi}')
    print('')

    previous_conda_env = None

    permissions_dirs = []
    activ_path = None

    for compiler, mpi in zip(compilers, mpis):

        python, recreate, conda_mpi, activ_suffix, env_suffix, \
            activ_path, conda_env_path, conda_env_name, activate_env, \
            spack_env = get_env_setup(args, config, machine, compiler, mpi,
                                      env_type, source_path, conda_base,
                                      args.env_name, compass_version, logger)

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

        if args.spack_base is not None:
            spack_base = args.spack_base
        elif e3sm_machine and compiler is not None:
            spack_base = get_spack_base(args.spack_base, config)
        else:
            spack_base = None

        if spack_base is not None and args.update_spack:
            # even if this is not a release, we need to update permissions on
            # shared system libraries
            permissions_dirs.append(spack_base)

        if previous_conda_env != conda_env_name:
            build_conda_env(
                env_type, recreate, machine, mpi, conda_mpi, compass_version,
                python, source_path, conda_template_path, conda_base,
                conda_env_name, conda_env_path, activate_base, args.use_local,
                args.local_conda_build, logger)
            previous_conda_env = conda_env_name

            if env_type != 'dev':
                permissions_dirs.append(conda_base)

        spack_script = ''
        if compiler is not None:
            env_vars = get_env_vars(machine, compiler, mpi)
            if spack_base is not None:
                spack_branch_base, spack_script, env_vars = build_spack_env(
                    config, args.update_spack, machine, compiler, mpi,
                    spack_env, spack_base, spack_template_path, env_vars,
                    args.tmpdir, logger)
                spack_script = f'echo Loading Spack environment...\n' \
                               f'{spack_script}\n' \
                               f'echo Done.\n' \
                               f'echo\n'
            else:
                env_vars = \
                    f'{env_vars}' \
                    f'export PIO={conda_env_path}\n' \
                    f'export OPENMP_INCLUDE=-I"{conda_env_path}/include"\n'
        else:
            env_vars = ''

        if env_type == 'dev':
            if args.env_name is not None:
                prefix = 'load_{}'.format(args.env_name)
            else:
                prefix = 'load_dev_compass_{}'.format(compass_version)
        elif env_type == 'test_release':
            prefix = 'test_compass_{}'.format(compass_version)
        else:
            prefix = 'load_compass_{}'.format(compass_version)

        script_filename = write_load_compass(
            conda_template_path, activ_path, conda_base, env_type,
            activ_suffix, prefix, conda_env_name, spack_script, machine,
            env_vars, args.env_only, source_path, args.without_openmp)

        if args.check:
            check_env(script_filename, conda_env_name, logger)

        if env_type == 'release' and not (args.with_albany or
                                          args.with_netlib_lapack or
                                          args.with_petsc):
            # make a symlink to the activation script
            link = os.path.join(activ_path,
                                f'load_latest_compass_{compiler}_{mpi}.sh')
            check_call(f'ln -sfn {script_filename} {link}')

            default_compiler = config.get('deploy', 'compiler')
            default_mpi = config.get('deploy',
                                     'mpi_{}'.format(default_compiler))
            if compiler == default_compiler and mpi == default_mpi:
                # make a default symlink to the activation script
                link = os.path.join(activ_path, 'load_latest_compass.sh')
                check_call(f'ln -sfn {script_filename} {link}')
        os.chdir(source_path)

    commands = '{}; conda clean -y -p -t'.format(activate_base)
    check_call(commands, logger=logger)

    if args.update_spack or env_type != 'dev':
        # we need to update permissions on shared stuff
        update_permissions(config, env_type, activ_path, permissions_dirs)


if __name__ == '__main__':
    main()
