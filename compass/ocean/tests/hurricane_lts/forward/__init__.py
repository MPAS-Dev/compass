import os

from compass.ocean.tests.hurricane_lts.analysis import Analysis
from compass.ocean.tests.hurricane_lts.configure import configure_hurricane_lts
from compass.ocean.tests.hurricane_lts.forward.forward import ForwardStep
from compass.testcase import TestCase


class Forward(TestCase):
    """
    A test case for performing a forward run for a hurricane
    case with local time-stepping

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.Mesh
        The test case that produces the mesh for this run

    init : compass.ocean.tests.hurricane_lts.init.Init
        The test case that produces the initial condition for this run
    """
    def __init__(self, test_group, mesh, storm, init):
        """
        Create test case

        Parameters
        ----------
        test_group : compass.ocean.tests.hurricane_lts.Hurricane_LTS
            The test group that this test case belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that produces the mesh for this run

        storm : str
            The name of the storm to be run

        init : compass.ocean.tests.hurricane_lts.init.Init
            The test case that produces the initial condition for this run
        """
        mesh_name = mesh.mesh_name
        subdir = os.path.join(mesh_name, storm)
        super().__init__(test_group=test_group,
                         subdir=subdir,
                         name=storm)
        self.mesh = mesh

        step = ForwardStep(test_case=self, mesh=mesh, init=init)

        step.add_output_file(filename='output.nc')
        step.add_output_file(filename='pointwiseStats.nc')
        self.add_step(step)

        step = Analysis(test_case=self, storm=storm)
        self.add_step(step)

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        configure_hurricane_lts(test_case=self, mesh=self.mesh)
