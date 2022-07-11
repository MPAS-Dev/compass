from compass.model import ModelStep


class ForwardStep(ModelStep):
    """
    A step for performing forward MPAS-Ocean runs as part of hurricane test
    cases.

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.Mesh
        The test case that produces the mesh for this run

    init : compass.ocean.tests.hurricane.init.Init
        The test case that produces the initial condition for this run
    """
    def __init__(self, test_case, mesh, init, name='forward', subdir=None):
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
        """
        self.mesh = mesh
        self.init = init
        super().__init__(test_case=test_case, name=name, openmp_threads=1)

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

    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies
        """
        self.ntasks = self.config.getint('hurricane', 'forward_ntasks')
        self.min_tasks = self.config.getint('hurricane', 'forward_min_tasks')
        self.openmp_threads = self.config.getint('hurricane',
                                                 'forward_threads')
