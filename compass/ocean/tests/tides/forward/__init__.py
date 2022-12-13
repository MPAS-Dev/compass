from compass.testcase import TestCase
from compass.ocean.tests.tides.forward.forward import ForwardStep
from compass.ocean.tests.tides.analysis import Analysis
from compass.ocean.tests.tides.configure import configure_tides
import os


class Forward(TestCase):
    """
    A test case for performing a forward run for a tidal case

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.Mesh
        The test case that produces the mesh for this run

    init : compass.ocean.tests.tides.init.Init
        The test case that produces the initial condition for this run
    """
    def __init__(self, test_group, mesh, init):
        """
        Create test case

        Parameters
        ----------
        test_group : compass.ocean.tests.tides.Hurricane
            The test group that this test case belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that produces the mesh for this run

        init : compass.ocean.tests.tides.init.Init
            The test case that produces the initial condition for this run
        """
        name = 'forward'
        mesh_name = mesh.mesh_name
        subdir = os.path.join(mesh_name, name)
        super().__init__(test_group=test_group,
                         subdir=subdir,
                         name=name)
        self.mesh = mesh

        step = ForwardStep(test_case=self, mesh=mesh, init=init)

        step.add_output_file(filename='output/output.nc')
        step.add_output_file(filename='analysis_members/harmonicAnalysis.nc')
        self.add_step(step)

        step = Analysis(test_case=self)
        self.add_step(step)

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        configure_tides(test_case=self, mesh=self.mesh)
