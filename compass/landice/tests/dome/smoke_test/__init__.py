from compass.testcase import TestCase
from compass.landice.tests.dome.setup_mesh import SetupMesh
from compass.landice.tests.dome.run_model import RunModel
from compass.landice.tests.dome.visualize import Visualize


class SmokeTest(TestCase):
    """
    The default test case for the dome test group simply creates the mesh and
    initial condition, then performs a short forward run on 4 cores.

    Attributes
    ----------
    mesh_type : str
        The resolution or tye of mesh of the test case
    """

    def __init__(self, test_group, mesh_type):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.dome.Dome
            The test group that this test case belongs to

        mesh_type : str
            The resolution or tye of mesh of the test case
        """
        name = 'smoke_test'
        self.mesh_type = mesh_type
        subdir = '{}/{}'.format(mesh_type, name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(
            SetupMesh(test_case=self, mesh_type=mesh_type))
        self.add_step(
            RunModel(test_case=self, cores=4, threads=1, mesh_type=mesh_type))
        step = Visualize(test_case=self, mesh_type=mesh_type)
        self.add_step(step, run_by_default=False)

    # no configure() method is needed because we will use the default dome
    # config options

    # no run() method is needed because we're doing the default: running all
    # steps
