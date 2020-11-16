import os
import multiprocessing
import subprocess


def get_available_cores_and_nodes(config):
    """
    Get the number of total cores and nodes available for running steps

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options for the testcase

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
        args = ['squeue', '--noheader', '-j', job_id, '-o', '"%C"']
        cores = subprocess.check_output(args)
        args = ['squeue', '--noheader', '-j', job_id, '-o', '"%D"']
        nodes = subprocess.check_output(args)
    elif parallel_system == 'single_node':
        cores = multiprocessing.cpu_count()
        nodes = 1
    else:
        raise ValueError('Unexpected parallel system: {}'.format(
            parallel_system))

    return cores, nodes


