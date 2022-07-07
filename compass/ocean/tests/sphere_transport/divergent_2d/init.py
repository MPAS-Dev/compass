from compass.model import ModelStep


class Init(ModelStep):
    """
    A step for an initial condition for for the cosine bell test case
    """

    def __init__(self, test_case, resolution):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.sphere_transport.divergent_2d.Divergent2D
            The test case this step belongs to

        resolution : int
            The resolution of the (uniform) mesh in km
        """

        super().__init__(test_case=test_case,
                         name=f'QU{resolution}_init',
                         subdir=f'QU{resolution}/init',
                         ntasks=36, min_tasks=1, openmp_threads=1)

        package = 'compass.ocean.tests.sphere_transport.divergent_2d'

        self.add_namelist_file(package, 'namelist.init', mode='init')
        self.add_streams_file(package, 'streams.init', mode='init')

        self.add_input_file(filename='mesh.nc', target='../mesh/mesh.nc')

        self.add_input_file(filename='graph.info', target='../mesh/graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='namelist.ocean')
        self.add_output_file(filename='initial_state.nc')
