from compass.model import run_model
from compass.step import Step


class ForwardStep(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of hurricane test
    cases.

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.Mesh
        The test case that produces the mesh for this run

    init : compass.ocean.tests.hurricane.init.Init
        The test case that produces the initial condition for this run

    cores_from_config : bool
        Whether to get ``cores`` from the config file

    min_cores_from_config : bool
        Whether to get ``min_cores`` from the config file

    threads_from_config : bool
        Whether to get ``threads`` from the config file
    """
    def __init__(self, test_case, mesh, init, name='forward',
                 subdir=None, cores=None, min_cores=None, threads=None):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.hurricane.forward.Forward
            The test case this step belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that produces the mesh for this run

        init : compass.ocean.tests.hurricane.init.Init
            The test case that produces the initial condition for this run

        name : str, optional
            the name of the step

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        cores : int, optional
            the number of cores the step would ideally use.  If fewer cores
            are available on the system, the step will run on all available
            cores as long as this is not below ``min_cores``

        min_cores : int, optional
            the number of cores the step requires.  If the system has fewer
            than this number of cores, the step will fail

        threads : int, optional
            the number of threads the step will use
        """
        self.mesh = mesh
        self.init = init
        if min_cores is None:
            min_cores = cores
        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         cores=cores, min_cores=min_cores, threads=threads)

        self.cores_from_config = cores is None
        self.min_cores_from_config = min_cores is None
        self.threads_from_config = threads is None

        self.add_namelist_file(
            'compass.ocean.tests.hurricane.forward', 'namelist.ocean')
        self.add_streams_file(
            'compass.ocean.tests.hurricane.forward', 'streams.ocean')

        mesh_package = mesh.mesh_step.package
        self.add_namelist_file(mesh_package, 'namelist.ocean')

        initial_state_target = \
            f'{init.path}/initial_state/ocean.nc'
        self.add_input_file(filename='input.nc',
                            work_dir_target=initial_state_target)
        self.add_input_file(
            filename='atmospheric_forcing.nc',
            work_dir_target=f'{init.path}/interpolate/atmospheric_forcing.nc')
        self.add_input_file(
            filename='points.nc',
            work_dir_target=f'{init.path}/pointstats/points.nc')
        self.add_input_file(
            filename='graph.info',
            work_dir_target=f'{init.path}/initial_state/graph.info')

        self.add_model_as_input()

    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies
        """
        if self.cores_from_config:
            self.cores = self.config.getint(
                'hurricane', 'forward_cores')
        if self.min_cores_from_config:
            self.min_cores = self.config.getint(
                'hurricane', 'forward_min_cores')
        if self.threads_from_config:
            self.threads = self.config.getint(
                'hurricane', 'forward_threads')

    def run(self):
        """
        Run this step of the testcase
        """
        run_model(self)
