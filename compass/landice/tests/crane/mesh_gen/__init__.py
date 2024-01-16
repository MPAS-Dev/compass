from compass.landice.tests.crane.mesh import Mesh
from compass.testcase import TestCase
from compass.validate import compare_variables


class MeshGen(TestCase):
    """
    The high resolution test case for the crane test
    group simply creates the mesh and initial condition.
    The basal friction optimization occurs separately,
    outside of COMPASS.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.crane.Crane
            The test group that this test case belongs to
        """
        name = 'mesh_gen'
        subdir = name
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(
            Mesh(test_case=self))

    def validate(self):
        """
        Compare ``thickness``, ``bedTopography``, ``iceMask`` and
        ``beta`` with a baseline if one was provided.
        """
        variables = ['thickness', 'bedTopography', 'iceMask',
                     'beta']
        compare_variables(test_case=self, variables=variables,
                          filename1='mesh/Crane.nc')
