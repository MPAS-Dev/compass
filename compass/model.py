import os
import xarray

from mpas_tools.logging import check_call
from compass.parallel import run_command

from compass.step import Step


class ModelStep(Step):
    """
    Attributes
    ----------

    namelist : str
        The name of the namelist file

    streams : str
        The name of the streams file

    update_pio : bool
        Whether to modify the namelist so the number of PIO tasks and the
        stride between them is consistent with the number of nodes and
        cores (one PIO task per node).

    make_graph : bool
        Whether to make a graph file from the given MPAS mesh file.  If
        ``True``, ``mesh_filename`` must be given.

    mesh_filename : str
        The name of an MPAS mesh file to use to make the graph file

    partition_graph : bool
        Whether to partition the domain for the requested number of cores.
        If so, the partitioning executable is taken from the ``partition``
        option of the ``[executables]`` config section.

    graph_filename : str
        The name of the graph file to partition

    """
    def __init__(self, test_case, name, subdir=None, ntasks=None,
                 min_tasks=None, openmp_threads=None, max_memory=None,
                 cached=False, namelist=None, streams=None, update_pio=True,
                 make_graph=False, mesh_filename=None, partition_graph=True,
                 graph_filename='graph.info'):
        """
        Make a step for running the model

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        name : str
            The name of the step

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        ntasks : int, optional
            the target number of tasks the step would ideally use.  If too
            few cores are available on the system to accommodate the number of
            tasks and the number of cores per task, the step will run on
            fewer tasks as long as as this is not below ``min_tasks``

        min_tasks : int, optional
            the number of tasks the step requires.  If the system has too
            few cores to accommodate the number of tasks and cores per task,
            the step will fail

        openmp_threads : int, optional
            the number of OpenMP threads to use

        max_memory : int, optional
            the amount of memory that the step is allowed to use in MB.
            This is currently just a placeholder for later use with task
            parallelism

        cached : bool, optional
            Whether to get all of the outputs for the step from the database of
            cached outputs for this MPAS core

        namelist : str, optional
            The name of the namelist file, default is ``namelist.<mpas_core>``

        streams : str, optional
            The name of the streams file, default is ``streams.<mpas_core>``

        update_pio : bool, optional
            Whether to modify the namelist so the number of PIO tasks and the
            stride between them is consistent with the number of nodes and
            cores (one PIO task per node).

        make_graph : bool, optional
            Whether to make a graph file from the given MPAS mesh file.  If
            ``True``, ``mesh_filename`` must be given.

        mesh_filename : str, optional
            The name of an MPAS mesh file to use to make the graph file

        partition_graph : bool, optional
            Whether to partition the domain for the requested number of cores.
            If so, the partitioning executable is taken from the ``partition``
            option of the ``[executables]`` config section.

        graph_filename : str, optional
            The name of the graph file to partition
        """
        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         cpus_per_task=openmp_threads,
                         min_cpus_per_task=openmp_threads, ntasks=ntasks,
                         min_tasks=min_tasks, openmp_threads=openmp_threads,
                         max_memory=max_memory, cached=cached)

        mpas_core = test_case.mpas_core.name
        if namelist is None:
            namelist = 'namelist.{}'.format(mpas_core)

        if streams is None:
            streams = 'streams.{}'.format(mpas_core)

        self.namelist = namelist
        self.streams = streams
        self.update_pio = update_pio
        self.make_graph = make_graph
        self.mesh_filename = mesh_filename
        self.partition_graph = partition_graph
        self.graph_filename = graph_filename

        self.add_input_file(filename='<<<model>>>')

    def setup(self):
        """ Setup the command-line arguments """
        config = self.config
        model = config.get('executables', 'model')
        model_basename = os.path.basename(model)
        self.args = [f'./{model_basename}', '-n', self.namelist,
                     '-s', self.streams]

    def set_model_resources(self, ntasks=None, min_tasks=None,
                            openmp_threads=None, max_memory=None):
        """
        Update the resources for the step.  This can be done within init,
        ``setup()`` or ``runtime_setup()`` for the step that this step
        belongs to, or init, ``configure()`` or ``run()`` for the test case
        that this step belongs to.
        Parameters
        ----------
        ntasks : int, optional
            the number of tasks the step would ideally use.  If too few
            cores are available on the system to accommodate the number of
            tasks and the number of cores per task, the step will run on
            fewer tasks as long as as this is not below ``min_tasks``

        min_tasks : int, optional
            the number of tasks the step requires.  If the system has too
            few cores to accommodate the number of tasks and cores per task,
            the step will fail

        openmp_threads : int, optional
            the number of OpenMP threads to use

        max_memory : int, optional
            the amount of memory that the step is allowed to use in MB.
            This is currently just a placeholder for later use with task
            parallelism
        """
        self.set_resources(cpus_per_task=openmp_threads,
                           min_cpus_per_task=openmp_threads, ntasks=ntasks,
                           min_tasks=min_tasks, openmp_threads=openmp_threads,
                           max_memory=max_memory)

    def runtime_setup(self):
        """
        Update PIO namelist options, make graph file, and partition graph file
        (if any of these are requested)
        """

        namelist = self.namelist
        cores = self.ntasks
        config = self.config
        logger = self.logger

        if self.update_pio:
            self.update_namelist_pio(namelist)

        if self.make_graph:
            make_graph_file(mesh_filename=self.mesh_filename,
                            graph_filename=self.graph_filename)

        if self.partition_graph:
            partition(cores, config, logger, graph_file=self.graph_filename)

    def process_inputs_and_outputs(self):
        """
        Process the model as an input, then call the parent class' version
        """
        for entry in self.input_data:
            filename = entry['filename']

            if filename == '<<<model>>>':
                model = self.config.get('executables', 'model')
                entry['filename'] = os.path.basename(model)
                entry['target'] = os.path.abspath(model)

        super().process_inputs_and_outputs()


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
