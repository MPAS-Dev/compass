import multiprocessing
import os
import subprocess
import warnings

from mpas_tools.logging import check_call


def get_available_parallel_resources(config):
    """
    Get the number of total cores and nodes available for running steps

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for the test case

    Returns
    -------
    available_resources : dict
        A dictionary containing available resources (cores, tasks, nodes
        and cores_per_node)
    """

    parallel_system = config.get('parallel', 'system')
    if parallel_system == 'slurm' and 'SLURM_JOB_ID' not in os.environ:
        parallel_system = 'login'

    if parallel_system == 'slurm':
        job_id = os.environ['SLURM_JOB_ID']
        node = os.environ['SLURMD_NODENAME']
        args = ['sinfo', '--noheader', '--node', node, '-o', '%X']
        sockets_per_node = _get_subprocess_int(args)
        args = ['sinfo', '--noheader', '--node', node, '-o', '%Y']
        cores_per_socket = _get_subprocess_int(args)
        if config.has_option('parallel', 'threads_per_core'):
            threads_per_core = config.getint('parallel', 'threads_per_core')
        else:
            args = ['sinfo', '--noheader', '--node', node, '-o', '%Z']
            threads_per_core = _get_subprocess_int(args)
        cores_per_node = sockets_per_node * cores_per_socket * threads_per_core
        args = ['squeue', '--noheader', '-j', job_id, '-o', '%D']
        nodes = _get_subprocess_int(args)
        cores = cores_per_node * nodes
        mpi_allowed = True
    elif parallel_system == 'login':
        cores = min(multiprocessing.cpu_count(),
                    config.getint('parallel', 'login_cores'))
        cores_per_node = cores
        nodes = 1
        mpi_allowed = False
    elif parallel_system == 'single_node':
        cores = multiprocessing.cpu_count()
        if config.has_option('parallel', 'cores_per_node'):
            cores = min(cores, config.getint('parallel', 'cores_per_node'))
        cores_per_node = cores
        nodes = 1
        mpi_allowed = True
    else:
        raise ValueError(f'Unexpected parallel system: {parallel_system}')

    available_resources = dict(
        cores=cores,
        nodes=nodes,
        cores_per_node=cores_per_node,
        mpi_allowed=mpi_allowed
    )
    return available_resources


def set_cores_per_node(config, cores_per_node):
    """
    If the system has Slurm, find out the ``cpus_per_node`` and set the config
    option accordingly.
    """
    parallel_system = config.get('parallel', 'system')
    if parallel_system == 'slurm':
        old_cores_per_node = config.getint('parallel', 'cores_per_node')
        config.set('parallel', 'cores_per_node', f'{cores_per_node}')
        if old_cores_per_node != cores_per_node:
            warnings.warn(f'Slurm found {cores_per_node} cpus per node but '
                          f'config from mache was {old_cores_per_node}')
    elif parallel_system == 'single_node':
        if not config.has_option('parallel', 'cores_per_node'):
            config.set('parallel', 'cores_per_node', f'{cores_per_node}')


def run_command(args, cpus_per_task, ntasks, openmp_threads, config, logger):
    """
    Run a subprocess with the given command-line arguments and resources

    Parameters
    ----------
    args : list of str
        The command-line arguments to run in parallel

    cpus_per_task : int
        the number of cores per task the process would ideally use.  If
        fewer cores per node are available on the system, the substep will
        run on all available cores as long as this is not below
        ``min_cpus_per_task``

    ntasks : int
        the number of tasks the process would ideally use.  If too few
        cores are available on the system to accommodate the number of
        tasks and the number of cores per task, the substep will run on
        fewer tasks as long as as this is not below ``min_tasks``

    openmp_threads : int
        the number of OpenMP threads to use

    config : configparser.ConfigParser
        Configuration options for the test case

    logger : logging.Logger
        A logger for output from the step
    """
    env = dict(os.environ)

    env['OMP_NUM_THREADS'] = f'{openmp_threads}'
    if openmp_threads > 1:
        logger.info(f'Running with {openmp_threads} OpenMP threads')

    parallel_executable = config.get('parallel', 'parallel_executable')

    # split the parallel executable into constituents in case it includes flags
    command_line_args = parallel_executable.split(' ')
    parallel_system = config.get('parallel', 'system')
    if parallel_system == 'slurm':
        command_line_args.extend(['-c', f'{cpus_per_task}', '-n', f'{ntasks}'])
    elif parallel_system == 'single_node':
        command_line_args.extend(['-n', f'{ntasks}'])
    else:
        raise ValueError(f'Unexpected parallel system: {parallel_system}')

    command_line_args.extend(args)

    check_call(command_line_args, logger, env=env)


def _get_subprocess_int(args):
    value = subprocess.check_output(args)
    value = int(value.decode('utf-8').strip('\n'))
    return value
