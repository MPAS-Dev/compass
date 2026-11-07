import os
from importlib import resources

import numpy as np
from jinja2 import Template


def write_custom_job_script(config, section, work_dir, job_name, ntasks):
    """
    Write ``job_script.custom.sh``, which runs MALI directly from the run
    directory with a user-supplied environment, if requested in the config

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for the test case

    section : str
        The config section with the ``custom_job_script`` and
        ``custom_job_env_cmds`` options

    work_dir : str
        The work directory where the job script should be written

    job_name : str
        The name of the batch job

    ntasks : int
        The number of MPI tasks for the model run
    """
    option = config.get(section, 'custom_job_script')
    if option == 'None':
        return
    if option != 'custom':
        raise ValueError(f"[{section}] custom_job_script must be 'None' or "
                         f"'custom', not '{option}'")

    env_cmds = config.get(section, 'custom_job_env_cmds')
    if env_cmds == 'None':
        raise ValueError(f"[{section}] custom_job_env_cmds must be set when "
                         f"custom_job_script = custom")

    if config.has_option('parallel', 'account'):
        account = config.get('parallel', 'account')
    else:
        account = ''

    if config.has_option('parallel', 'constraints'):
        constraint = config.getlist('parallel', 'constraints')[0]
    else:
        constraint = ''

    cores_per_node = config.getint('parallel', 'cores_per_node')
    nodes = int(np.ceil(ntasks / cores_per_node))

    model = os.path.basename(config.get('executables', 'model'))

    template = Template(resources.read_text(
        'compass.landice.tests.ismip7_run', 'job_script_custom.template'))
    text = template.render(
        job_name=job_name, account=account, nodes=f'{nodes}',
        constraint=constraint, env_cmds=env_cmds, ntasks=ntasks, model=model,
        parallel_executable=config.get('parallel', 'parallel_executable'))

    with open(os.path.join(work_dir, 'job_script.custom.sh'), 'w') as handle:
        handle.write(text)
