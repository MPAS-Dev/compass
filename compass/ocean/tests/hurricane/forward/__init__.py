from compass.testcase import TestCase
from compass.ocean.tests.hurricane.forward.forward import ForwardStep
from compass.ocean.tests.hurricane.analysis import Analysis 
from compass.ocean.tests.global_ocean.configure import configure_global_ocean
import os

class Forward(TestCase):
    """
    A test case for performing two forward run, one without a restart and one
    with to make sure the results are identical
    """

    def __init__(self, test_group, mesh, storm, init):
        """
        Create test case

        Parameters
        ----------
        test_group : compass.ocean.tests.global_ocean.GlobalOcean
            The global ocean test group that this test case belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that produces the mesh for this run

        init : compass.ocean.tests.global_ocean.init.Init
            The test case that produces the initial condition for this run

        time_integrator : {'split_explicit', 'RK4'}
            The time integrator to use for the forward run
        """

        mesh_name = mesh.mesh_name
        subdir = os.path.join(mesh_name, storm)
        super().__init__(test_group=test_group,
                         subdir=subdir,
                         name=storm)
        self.mesh = mesh
        self.init = init

        step = ForwardStep(test_case=self, mesh=mesh, init=init)

        step.add_output_file(filename='output.nc')  
        step.add_output_file(filename='pointwiseStats.nc')  
        self.add_step(step)

        step = Analysis(test_case=self,storm=storm)
        self.add_step(step)

    def configure(self):
        """ 
        Modify the configuration options for this test case
        """
        configure_global_ocean(test_case=self, mesh=self.mesh, init=self.init)

    def run(self):
        """ 
        Run each step of the testcase
        """

        # run the steps
        super().run()
