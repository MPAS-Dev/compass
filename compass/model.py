import os
import xarray

from mpas_tools.logging import check_call
from compass.parallel import run_command


def run_model(step, update_pio=True, partition_graph=True,
              graph_file='graph.info', namelist=None, streams=None):
    """
    Run the model after determining the number of tasks and threads

    Parameters
    ----------
    step : compass.Step
        a step

    update_pio : bool, optional
        Whether to modify the namelist so the number of PIO tasks and the
        stride between them is consistent with the number of nodes and tasks
        (one PIO task per node).

    partition_graph : bool, optional
        Whether to partition the domain for the requested number of tasks.  If
        so, the partitioning executable is taken from the ``partition`` option
        of the ``[executables]`` config section.

    graph_file : str, optional
        The name of the graph file to partition

    namelist : str, optional
        The name of the namelist file, default is ``namelist.<mpas_core>``

    streams : str, optional
        The name of the streams file, default is ``streams.<mpas_core>``
    """
    mpas_core = step.mpas_core.name
    ntasks = step.ntasks
    cpus_per_task = step.cpus_per_task
    openmp_threads = step.openmp_threads
    config = step.config
    logger = step.logger

    if namelist is None:
        namelist = f'namelist.{mpas_core}'

    if streams is None:
        streams = f'streams.{mpas_core}'

    if update_pio:
        step.update_namelist_pio(namelist)

    if partition_graph:
        partition(ntasks, config, logger, graph_file=graph_file)

    model = config.get('executables', 'model')
    model_basename = os.path.basename(model)
    args = [f'./{model_basename}', '-n', namelist, '-s', streams]
    run_command(args=args, cpus_per_task=cpus_per_task, ntasks=ntasks,
                openmp_threads=openmp_threads, config=config, logger=logger)


def partition(ntasks, config, logger, graph_file='graph.info'):
    """
    Partition the domain for the requested number of tasks

    Parameters
    ----------
    ntasks : int
        The number of tasks that the model should be run on

    config : compass.config.CompassConfigParser
        Configuration options for the test case, used to get the partitioning
        executable

    logger : logging.Logger
        A logger for output from the step that is calling this function

    graph_file : str, optional
        The name of the graph file to partition

    """
    if ntasks > 1:
        executable = config.get('parallel', 'partition_executable')
        args = [executable, graph_file, f'{ntasks}']
        check_call(args, logger)


def make_graph_file(mesh_filename, graph_filename='graph.info',
                    weight_field=None):
    """
    Make a graph file from the MPAS mesh for use in the Metis graph
    partitioning software

    Parameters
    ----------
     mesh_filename : str
        The name of the input MPAS mesh file

    graph_filename : str, optional
        The name of the output graph file

    weight_field : str
        The name of a variable in the MPAS mesh file to use as a field of
        weights
    """

    with xarray.open_dataset(mesh_filename) as ds:

        nCells = ds.sizes['nCells']

        nEdgesOnCell = ds.nEdgesOnCell.values
        cellsOnCell = ds.cellsOnCell.values - 1
        if weight_field is not None:
            if weight_field in ds:
                raise ValueError('weight_field {} not found in {}'.format(
                    weight_field, mesh_filename))
            weights = ds[weight_field].values
        else:
            weights = None

    nEdges = 0
    for i in range(nCells):
        for j in range(nEdgesOnCell[i]):
            if cellsOnCell[i][j] != -1:
                nEdges = nEdges + 1

    nEdges = nEdges/2

    with open(graph_filename, 'w+') as graph:
        if weights is None:
            graph.write('{} {}\n'.format(nCells, nEdges))

            for i in range(nCells):
                for j in range(0, nEdgesOnCell[i]):
                    if cellsOnCell[i][j] >= 0:
                        graph.write('{} '.format(cellsOnCell[i][j]+1))
                graph.write('\n')
        else:
            graph.write('{} {} 010\n'.format(nCells, nEdges))

            for i in range(nCells):
                graph.write('{} '.format(int(weights[i])))
                for j in range(0, nEdgesOnCell[i]):
                    if cellsOnCell[i][j] >= 0:
                        graph.write('{} '.format(cellsOnCell[i][j] + 1))
                graph.write('\n')
