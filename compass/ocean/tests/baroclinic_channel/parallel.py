import numpy

from compass.parallel import get_available_cores_and_nodes
from compass import namelist


def get_core_count(config, target_cores, step_dir):
    """
    Determine an appropriate number of cores for MPAS-Ocean to run with, based
    on the resolution and the available node and core counts.

    Modify the namelist so the number of PIO tasks and the stride between them
    is consistent with the number of nodes and cores (one PIO task per node).

    Parameters
    ----------
     config : configparser.ConfigParser
        Configuration options for this testcase

    target_cores : int
        The number of cores requested

    step_dir : str
        The work directory for this step of the testcase

    Returns
    -------
    core_count : int
        The number of cores to use in this step of the testcase
    """

    cores_per_node = config.getint('parallel', 'cores_per_node')
    available_cores, available_nodes = get_available_cores_and_nodes(config)
    core_count = min(target_cores, available_cores)

    # update PIO tasks based on the machine settings and the available number
    # or cores
    pio_num_iotasks = int(numpy.ceil(core_count/cores_per_node))
    pio_stride = min(cores_per_node, core_count)

    replacements = {'config_pio_num_iotasks': '{}'.format(pio_num_iotasks),
                    'config_pio_stride': '{}'.format(pio_stride)}

    namelist.update(replacements=replacements, step_work_dir=step_dir,
                    core='ocean')

    return core_count
