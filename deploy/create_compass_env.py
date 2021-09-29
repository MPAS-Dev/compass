#!/usr/bin/env python
import subprocess
import os
import socket
import glob
import stat
import grp
import requests
import progressbar


def get_envs():
    # Modify the following list of dictionaries to choose which compass
    # version, python version, and which mpi variant (nompi, mpich or openmpi)
    # to use.

    envs = [{'suffix': '',
             'version': '0.3.0',
             'python': '3.9',
             'mpi': 'nompi'}]

    # whether to delete and rebuild each environment if it already exists
    force_recreate = True

    # whether these are to be test environments
    is_test = False

    return envs, force_recreate, is_test


def get_host_info():
    hostname = socket.gethostname()
    if hostname.startswith('cori') or hostname.startswith('dtn'):
        base_path = "/global/cfs/cdirs/e3sm/software/anaconda_envs/base"
        activ_path = "/global/cfs/cdirs/e3sm/software/anaconda_envs"
        group = "e3sm"
    elif hostname.startswith('acme1') or hostname.startswith('aims4'):
        base_path = "/usr/local/e3sm_unified/envs/base"
        activ_path = "/usr/local/e3sm_unified/envs"
        group = "climate"
    elif hostname.startswith('blueslogin'):
        base_path = "/lcrc/soft/climate/e3sm-unified/base"
        activ_path = "/lcrc/soft/climate/e3sm-unified"
        group = "cels"
    elif hostname.startswith('rhea'):
        base_path = "/ccs/proj/cli900/sw/rhea/e3sm-unified/base"
        activ_path = "/ccs/proj/cli900/sw/rhea/e3sm-unified"
        group = "cli900"
    elif hostname.startswith('cooley'):
        base_path = "/lus/theta-fs0/projects/ccsm/acme/tools/e3sm-unified/base"
        activ_path = "/lus/theta-fs0/projects/ccsm/acme/tools/e3sm-unified"
        group = "ccsm"
    elif hostname.startswith('compy'):
        base_path = "/share/apps/E3SM/conda_envs/base"
        activ_path = "/share/apps/E3SM/conda_envs"
        group = "users"
    elif hostname.startswith('gr-fe') or hostname.startswith('wf-fe'):
        base_path = "/usr/projects/climate/SHARED_CLIMATE/anaconda_envs/base"
        activ_path = "/usr/projects/climate/SHARED_CLIMATE/anaconda_envs"
        group = "climate"
    elif hostname.startswith('burnham'):
        base_path = "/home/xylar/Desktop/test_compass/base"
        activ_path = "/home/xylar/Desktop/test_compass"
        group = "xylar"
    else:
        raise ValueError(
            "Unknown host name {}.  Add env_path and group for "
            "this machine to the script.".format(hostname))

    return base_path, activ_path, group


def check_env(base_path, env_name, env):
    print("Checking the environment {}".format(env_name))

    activate = 'source {}/etc/profile.d/conda.sh; conda activate {}'.format(
        base_path, env_name)

    imports = ['geometric_features', 'mpas_tools', 'jigsawpy']
    for import_name in imports:
        command = '{}; python -c "import {}"'.format(activate, import_name)
        test_command(command, os.environ, import_name)

    commands = [['gpmetis', '--help'],
                ['ffmpeg', '--help']]

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
    envs, force_recreate, is_test = get_envs()

    base_path, activ_path, group = get_host_info()

    if not os.path.exists(base_path):
        miniconda = 'Miniconda3-latest-Linux-x86_64.sh'
        url = 'https://repo.continuum.io/miniconda/{}'.format(miniconda)
        r = requests.get(url)
        with open(miniconda, 'wb') as outfile:
            outfile.write(r.content)

        command = '/bin/bash {} -b -p {}'.format(miniconda, base_path)
        subprocess.check_call(command, executable='/bin/bash', shell=True)
        os.remove(miniconda)

    print('Doing initial setup')
    activate = 'source {}/etc/profile.d/conda.sh; conda activate'.format(
        base_path)

    commands = '{}; conda config --add channels conda-forge; ' \
               'conda config --set channel_priority strict; ' \
               'conda update -y --all'.format(activate)

    subprocess.check_call(commands, executable='/bin/bash', shell=True)
    print('done')

    for env in envs:
        version = env['version']
        suffix = env['suffix']
        python = env['python']
        mpi = env['mpi']
        if mpi == 'nompi':
            mpi_prefix = 'nompi'
        else:
            mpi_prefix = 'mpi_{}'.format(mpi)

        channels = '--override-channels -c conda-forge -c defaults -c e3sm'
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

        check_env(base_path, env_name, env)

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
