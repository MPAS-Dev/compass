import os
import multiprocessing
import subprocess


def get_available_cores_and_nodes(config):
    """
    Get the number of total cores and nodes available for running steps

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options for the test case

    Returns
    -------
    cores : int
        The number of cores available for running steps

    nodes : int
        The number of cores available for running steps
    """

    parallel_system = config.get('parallel', 'system')
    if parallel_system == 'slurm':
        job_id = os.environ['SLURM_JOB_ID']
        args = ['squeue', '--noheader', '-j', job_id, '-o', '%C']
        cores = _get_subprocess_int(args)
        args = ['squeue', '--noheader', '-j', job_id, '-o', '%D']
        nodes = _get_subprocess_int(args)
    elif parallel_system == 'single_node':
        cores_per_node = config.getint('parallel', 'cores_per_node')
        cores = min(multiprocessing.cpu_count(), cores_per_node)
        nodes = 1
    else:
        raise ValueError('Unexpected parallel system: {}'.format(
            parallel_system))

    return cores, nodes


def _get_subprocess_int(args):
    value = subprocess.check_output(args)
    value = int(value.decode('utf-8').strip('\n'))
    return value
