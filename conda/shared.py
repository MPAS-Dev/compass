from __future__ import print_function

import os
import sys
import argparse
import subprocess
import platform
import logging
import shutil

try:
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request


def parse_args(bootstrap):
    parser = argparse.ArgumentParser(
        description='Deploy a compass conda environment')
    parser.add_argument("-m", "--machine", dest="machine",
                        help="The name of the machine for loading machine-"
                             "related config options")
    parser.add_argument("--conda", dest="conda_base",
                        help="Path to the conda base")
    parser.add_argument("--spack", dest="spack_base",
                        help="Path to the spack base")
    parser.add_argument("--env_name", dest="env_name",
                        help="The conda environment name and activation script"
                             " prefix")
    parser.add_argument("-p", "--python", dest="python", type=str,
                        help="The python version to deploy")
    parser.add_argument("-c", "--compiler", dest="compilers", type=str,
                        nargs="*", help="The name of the compiler(s)")
    parser.add_argument("-i", "--mpi", dest="mpis", type=str, nargs="*",
                        help="The MPI library (or libraries) to deploy, see "
                             "the docs for details")
    parser.add_argument("--env_only", dest="env_only", action='store_true',
                        help="Create only the compass environment, don't "
                             "install compilers or build SCORPIO")
    parser.add_argument("--recreate", dest="recreate", action='store_true',
                        help="Recreate the environment if it exists")
    parser.add_argument("-f", "--config_file", dest="config_file",
                        help="Config file to override deployment config "
                             "options")
    parser.add_argument("--check", dest="check", action='store_true',
                        help="Check the resulting environment for expected "
                             "packages")
    parser.add_argument("--use_local", dest="use_local", action='store_true',
                        help="Use locally built conda packages (for testing).")
    parser.add_argument("--update_spack", dest="update_spack",
                        action='store_true',
                        help="If the shared spack environment should be "
                             "created or recreated.")
    parser.add_argument("--tmpdir", dest="tmpdir",
                        help="A temporary directory for building spack "
                             "packages")
    parser.add_argument("--with_albany", dest="with_albany",
                        action='store_true',
                        help="Whether to include albany in the spack "
                             "environment")
    parser.add_argument("--with_netlib_lapack", dest="with_netlib_lapack",
                        action='store_true',
                        help="Whether to include Netlib-LAPACK in the spack "
                             "environment")
    parser.add_argument("--with_petsc", dest="with_petsc",
                        action='store_true',
                        help="Whether to include PETSc in the spack "
                             "environment")
    parser.add_argument("--without_openmp", dest="without_openmp",
                        action='store_true',
                        help="If this flag is included, OPENMP=false will "
                             "be added to the load script.  By default, MPAS "
                             "builds will be with OpenMP (OPENMP=true).")
    if bootstrap:
        parser.add_argument("--local_conda_build", dest="local_conda_build",
                            type=str,
                            help="A path for conda packages (for testing).")

    args = parser.parse_args(sys.argv[1:])

    return args


def get_conda_base(conda_base, config, shared=False, warn=False):
    if shared:
        conda_base = config.get('paths', 'compass_envs')
    elif conda_base is None:
        if 'CONDA_EXE' in os.environ:
            # if this is a test, assume we're the same base as the
            # environment currently active
            conda_exe = os.environ['CONDA_EXE']
            conda_base = os.path.abspath(
                os.path.join(conda_exe, '..', '..'))
            if warn:
                print('\nWarning: --conda path not supplied.  Using conda '
                      'installed at:\n'
                      '   {}\n'.format(conda_base))
        else:
            raise ValueError('No conda base provided with --conda and '
                             'none could be inferred.')
    # handle "~" in the path
    conda_base = os.path.abspath(os.path.expanduser(conda_base))
    return conda_base


def get_spack_base(spack_base, config):
    if spack_base is None:
        if config.has_option('deploy', 'spack'):
            spack_base = config.get('deploy', 'spack')
        else:
            raise ValueError('No spack base provided with --spack and none is '
                             'provided in a config file.')
    # handle "~" in the path
    spack_base = os.path.abspath(os.path.expanduser(spack_base))
    return spack_base


def check_call(commands, env=None, logger=None):
    print_command = '\n   '.join(commands.split('; '))
    if logger is None:
        print('\n Running:\n   {}\n'.format(print_command))
    else:
        logger.info('\nrunning:\n   {}\n'.format(print_command))

    if logger is None:
        process = subprocess.Popen(commands, env=env, executable='/bin/bash',
                                   shell=True)
        process.wait()
    else:
        process = subprocess.Popen(commands, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=env,
                                   executable='/bin/bash', shell=True)
        stdout, stderr = process.communicate()

        if stdout:
            stdout = stdout.decode('utf-8')
            for line in stdout.split('\n'):
                logger.info(line)
        if stderr:
            stderr = stderr.decode('utf-8')
            for line in stderr.split('\n'):
                logger.error(line)

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, commands)


def install_miniconda(conda_base, activate_base, logger):
    if not os.path.exists(conda_base):
        print('Installing Mambaforge')
        if platform.system() == 'Linux':
            system = 'Linux'
        elif platform.system() == 'Darwin':
            system = 'MacOSX'
        else:
            system = 'Linux'
        miniconda = 'Mambaforge-{}-x86_64.sh'.format(system)
        url = 'https://github.com/conda-forge/miniforge/releases/latest/download/{}'.format(miniconda)
        print(url)
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        f = urlopen(req)
        html = f.read()
        with open(miniconda, 'wb') as outfile:
            outfile.write(html)
        f.close()

        command = '/bin/bash {} -b -p {}'.format(miniconda, conda_base)
        check_call(command, logger=logger)
        os.remove(miniconda)

    backup_bashrc()

    print('Doing initial setup\n')
    commands = '{}; ' \
               'conda config --add channels conda-forge; ' \
               'conda config --set channel_priority strict; ' \
               'conda install -y boa; ' \
               'conda update -y --all; ' \
               'mamba init'.format(activate_base)

    check_call(commands, logger=logger)

    restore_bashrc()


def backup_bashrc():
    home_dir = os.path.expanduser('~')
    files = ['.bashrc', '.bash_profile']
    for filename in files:
        src = os.path.join(home_dir, filename)
        dst = os.path.join(home_dir, '{}.conda_bak'.format(filename))
        if os.path.exists(src):
            shutil.copyfile(src, dst)


def restore_bashrc():
    home_dir = os.path.expanduser('~')
    files = ['.bashrc', '.bash_profile']
    for filename in files:
        src = os.path.join(home_dir, '{}.conda_bak'.format(filename))
        dst = os.path.join(home_dir, filename)
        if os.path.exists(src):
            shutil.move(src, dst)


def get_logger(name, log_filename):
    print('Logging to: {}\n'.format(log_filename))
    try:
        os.remove(log_filename)
    except OSError:
        pass
    logger = logging.getLogger(name)
    handler = logging.FileHandler(log_filename)
    formatter = CompassFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class CompassFormatter(logging.Formatter):
    """
    A custom formatter for logging
    Modified from:
    https://stackoverflow.com/a/8349076/7728169
    https://stackoverflow.com/a/14859558/7728169
    """

    # printing error messages without a prefix because they are sometimes
    # errors and sometimes only warnings sent to stderr
    dbg_fmt = "DEBUG: %(module)s: %(lineno)d: %(msg)s"
    info_fmt = "%(msg)s"
    err_fmt = info_fmt

    def __init__(self, fmt=info_fmt):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._fmt = CompassFormatter.dbg_fmt

        elif record.levelno == logging.INFO:
            self._fmt = CompassFormatter.info_fmt

        elif record.levelno == logging.ERROR:
            self._fmt = CompassFormatter.err_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._fmt = format_orig

        return result
