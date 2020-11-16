import os
import subprocess

from compass.io import symlink


def symlink_model(config, step_dir):
    """
    make a link to the model executable

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options for the testcase

    step_dir : str
        The work directory for the step where the model link should be added
    """
    model = config.get('executables', 'model')
    model_basename = os.path.basename(model)
    symlink(os.path.abspath(model), os.path.join(step_dir, model_basename))


def partition(core_count, executable='gpmetis', graph_file='graph.info'):
    """
    Partition the domain for the requested number of cores

    Parameters
    ----------
    core_count : int
        The number of cores that the model should be run on

    executable : str, optional
        The command to use for partitioning

    graph_file : str, optional
        The name of the graph file to partition

    """
    args = [executable, graph_file, '{}'.format(core_count)]
    subprocess.check_call(args)


def run_model(config, core, core_count, threads=1):
    """
    Run the model after determining the number of cores

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options for the testcase

    core : str
        The name of the MPAS core ('ocean', 'landice', etc.)

    core_count : int
        The maximum number of cores that the model should be run on

    threads : int
        The number of threads to use for the model run
    """
    os.environ['OMP_NUM_THREADS'] = '{}'.format(threads)

    namelist = 'namelist.{}'.format(core)

    streams = 'streams.{}'.format(core)

    parallel_executable = config.get('parallel', 'parallel_executable')
    model = config.get('executables', 'model')
    model_basename = os.path.basename(model)

    args = [parallel_executable,
            '-n', '{}'.format(core_count),
            './{}'.format(model_basename),
            '-n', namelist,
            '-s', streams]

    subprocess.check_call(args)
