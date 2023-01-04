from jinja2 import Template
from importlib import resources
import os
import numpy as np


def write_step_job_script(config, run_name, target_cores, min_cores, work_dir):
    """

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for this test case, a combination of user configs
        and the defaults for the machine and MPAS core

    run_name : str
        Name of this run to be used in job script

    target_cores : int
        The target number of cores for the job to use

    min_cores : int
        The minimum number of cores for the job to use

    work_dir : str
        The work directory where the job script should be written
    """

    if config.has_option('parallel', 'account'):
        account = config.get('parallel', 'account')
    else:
        account = ''

    cores_per_node = config.getint('parallel', 'cores_per_node')

    # as a rule of thumb, let's do the geometric mean between min and target
    cores = np.sqrt(target_cores*min_cores)
    nodes = int(np.ceil(cores/cores_per_node))

    machine = None # NEED TO FIX THIS
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

    job_name = config.get('job', 'job_name')
    if job_name == '<<<default>>>':
        job_name = run_name
    wall_time = config.get('job', 'wall_time')

    template = Template(resources.read_text(
        'compass.landice.tests.thwaites.uq_ensemble',
        'job_script.template'))

    text = template.render(job_name=job_name, account=account,
                           nodes=f'{nodes}', wall_time=wall_time, qos=qos,
                           partition=partition, constraint=constraint)
    text = _clean_up_whitespace(text)
    script_filename = 'job_script.sh'
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
