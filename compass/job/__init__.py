import os
from importlib import resources

import numpy as np
from jinja2 import Template


def write_job_script(config, machine, target_cores, min_cores, work_dir,
                     suite='', pre_run_commands='', post_run_commands=''):
    """

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for this test case, a combination of user configs
        and the defaults for the machine and MPAS core

    machine : {str, None}
        The name of the machine

    target_cores : int
        The target number of cores for the job to use

    min_cores : int
        The minimum number of cores for the job to use

    work_dir : str
        The work directory where the job script should be written

    suite : str, optional
        The name of the suite

    pre_run_commands : str, optional
        Optional commands to be inserted into job script prior to calling
        compass run

    post_run_commands : str, optional
        Optional commands to be inserted into job script after calling of
        compass run
    """

    if config.has_option('parallel', 'account'):
        account = config.get('parallel', 'account')
    else:
        account = ''

    cores_per_node = config.getint('parallel', 'cores_per_node')

    # as a rule of thumb, let's do the geometric mean between min and target
    cores = np.sqrt(target_cores * min_cores)
    nodes = int(np.ceil(cores / cores_per_node))

    partition = config.get('job', 'partition')
    if partition == '<<<default>>>':
        if machine == 'anvil':
            # choose the partition based on the number of nodes
            if nodes <= 5:
                partition = 'acme-small'
            elif nodes <= 60:
                partition = 'acme-medium'
            else:
                partition = 'acme-large'
        elif config.has_option('parallel', 'partitions'):
            # get the first, which is the default
            partition = config.getlist('parallel', 'partitions')[0]
        else:
            partition = ''

    qos = config.get('job', 'qos')
    if qos == '<<<default>>>':
        if config.has_option('parallel', 'qos'):
            # get the first, which is the default
            qos = config.getlist('parallel', 'qos')[0]
        else:
            qos = ''

    constraint = config.get('job', 'constraint')
    if constraint == '<<<default>>>':
        if config.has_option('parallel', 'constraints'):
            # get the first, which is the default
            constraint = config.getlist('parallel', 'constraints')[0]
        else:
            constraint = ''

    if config.has_option('job', 'reservation'):
        reservation = config.get('job', 'reservation')
    else:
        reservation = ''

    job_name = config.get('job', 'job_name')
    if job_name == '<<<default>>>':
        if suite == '':
            job_name = 'compass'
        else:
            job_name = f'compass_{suite}'
    wall_time = config.get('job', 'wall_time')

    template = Template(resources.read_text(
        'compass.job', 'job_script.template'))

    text = template.render(job_name=job_name, account=account,
                           nodes=f'{nodes}', wall_time=wall_time, qos=qos,
                           partition=partition, constraint=constraint,
                           reservation=reservation, suite=suite,
                           pre_run_commands=pre_run_commands,
                           post_run_commands=post_run_commands)
    text = _clean_up_whitespace(text)
    if suite == '':
        script_filename = 'job_script.sh'
    else:
        script_filename = f'job_script.{suite}.sh'
    script_filename = os.path.join(work_dir, script_filename)
    with open(script_filename, 'w') as handle:
        handle.write(text)


def _clean_up_whitespace(text):
    prev_line = None
    lines = text.split('\n')
    trimmed = list()
    # remove extra blank lines
    for line in lines:
        if line != '' or prev_line != '':
            trimmed.append(line)
            prev_line = line

    line = ''
    lines = list()
    # remove blank lines between comments
    for next_line in trimmed:
        if line != '' or not next_line.startswith('#'):
            lines.append(line)
        line = next_line

    # add the last line that we missed and an extra blank line
    lines.extend([trimmed[-1], ''])
    text = '\n'.join(lines)
    return text
