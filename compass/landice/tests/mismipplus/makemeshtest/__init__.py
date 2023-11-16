from compass.landice.tests.mismipplus.setup_mesh import SetupMesh
from compass.testcase import TestCase


class MakeMeshTest(TestCase):
    """
    """
    def __init__(self, test_group):

        name = "MakeMeshTest"

        super().__init__(test_group=test_group, name=name)

        self.add_step(SetupMesh(test_case=self))
