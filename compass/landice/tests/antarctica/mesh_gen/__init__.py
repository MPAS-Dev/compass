from compass.landice.tests.antarctica.mesh import Mesh
from compass.testcase import TestCase


class MeshGen(TestCase):
    """
    The default test case for the Antarctica test group simply creates the
    mesh and initial condition.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.antarctica.Antarctica
            The test group that this test case belongs to
        """
        name = 'mesh_gen'
        subdir = name
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(
            Mesh(test_case=self))
