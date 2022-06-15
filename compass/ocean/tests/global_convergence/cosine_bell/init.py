from compass.model import run_model
from compass.step import Step


class Init(Step):
    """
    A step for an initial condition for for the cosine bell test case
    """
    def __init__(self, test_case, mesh_name):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_convergence.cosine_bell.CosineBell
            The test case this step belongs to

        mesh_name : str
            The name of the mesh
        """

        super().__init__(test_case=test_case,
                         name=f'{mesh_name}_init',
                         subdir=f'{mesh_name}/init',
                         ntasks=36, min_tasks=1)

        package = 'compass.ocean.tests.global_convergence.cosine_bell'

        self.add_namelist_file(package, 'namelist.init', mode='init')
        self.add_streams_file(package, 'streams.init', mode='init')

        self.add_input_file(filename='mesh.nc', target='../mesh/mesh.nc')

        self.add_input_file(filename='graph.info', target='../mesh/graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='namelist.ocean')
        self.add_output_file(filename='initial_state.nc')

    def run(self):
        """
        Run this step of the testcase
        """
        run_model(self)
