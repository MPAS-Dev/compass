import os
import multiprocessing
import subprocess
import numpy

from compass import namelist, streams


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
        args = ['squeue', '--noheader', '-j', job_id, '-o', '%C']
        cores = _get_subprocess_int(args)
        args = ['squeue', '--noheader', '-j', job_id, '-o', '%D']
        nodes = _get_subprocess_int(args)
    elif parallel_system == 'single_node':
        cores = multiprocessing.cpu_count()
        nodes = 1
    else:
        raise ValueError('Unexpected parallel system: {}'.format(
            parallel_system))

    return cores, nodes


def update_namelist_pio(config, cores, step_dir):
    """
    Modify the namelist so the number of PIO tasks and the stride between them
    is consistent with the number of nodes and cores (one PIO task per node).

    Parameters
    ----------
     config : configparser.ConfigParser
        Configuration options for this testcase

    cores : int
        The number of cores

    step_dir : str
        The work directory for this step of the testcase
    """

    cores_per_node = config.getint('parallel', 'cores_per_node')

    # update PIO tasks based on the machine settings and the available number
    # or cores
    pio_num_iotasks = int(numpy.ceil(cores/cores_per_node))
    pio_stride = cores//pio_num_iotasks
    if pio_stride > cores_per_node:
        raise ValueError('Not enough nodes for the number of cores.  cores: '
                         '{}, cores per node: {}'.format(cores,
                                                         cores_per_node))

    replacements = {'config_pio_num_iotasks': '{}'.format(pio_num_iotasks),
                    'config_pio_stride': '{}'.format(pio_stride)}

    namelist.update(replacements=replacements, step_work_dir=step_dir,
                    core='ocean')


def _get_subprocess_int(args):
    value = subprocess.check_output(args)
    value = int(value.decode('utf-8').strip('\n'))
    return value
