from compass.model import run_model
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for tidal
    test cases

    Attributes
    ----------
    mesh : compass.ocean.tests.tides.mesh.mesh.MeshStep
        The step for creating the mesh

    """
    def __init__(self, test_case, mesh):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.tides.init.Init
            The test case this step belongs to

        mesh : compass.ocean.tests.tides.mesh.Mesh
            The test case that creates the mesh used by this test case

        """

        super().__init__(test_case=test_case, name='initial_state')
        self.mesh = mesh

        package = 'compass.ocean.tests.tides.init'

        # generate the namelist, replacing a few default options
        self.add_namelist_file(package, 'namelist.init', mode='init')

        # generate the streams file
        self.add_streams_file(package, 'streams.init', mode='init')

        mesh_path = mesh.steps['cull_mesh'].path

        self.add_input_file(
            filename='mesh.nc',
            work_dir_target=f'{mesh_path}/culled_mesh.nc')

        self.add_input_file(
            filename='graph.info',
            work_dir_target=f'{mesh_path}/culled_graph.info')

        self.add_model_as_input()

        for file in ['ocean.nc', 'graph.info']:
            self.add_output_file(filename=file)

    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies.
        """
        # get the these properties from the config options
        config = self.config
        self.cores = config.getint('tides', 'init_cores')
        self.min_cores = config.getint('tides', 'init_min_cores')
        self.threads = config.getint('tides', 'init_threads')

    def run(self):
        """
        Run this step of the testcase
        """
        run_model(self)
