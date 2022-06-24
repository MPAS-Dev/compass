from compass.testcase import TestCase
from compass.ocean.tests.gotm.default.mesh import Mesh
from compass.ocean.tests.gotm.default.init import Init
from compass.ocean.tests.gotm.default.forward import Forward
from compass.ocean.tests.gotm.default.analysis import Analysis
from compass.validate import compare_variables


class Default(TestCase):
    """
    The default test case for the General Ocean Turbulence Model (GOTM) test
    group creates an initial condition on a 4 x 4 cell, doubly periodic grid,
    performs a short simulation, then vertical plots of the velocity and
    viscosity.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.gotm.Gotm
            The test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='default')

        self.add_step(Mesh(test_case=self))
        self.add_step(Init(test_case=self))
        self.add_step(Forward(test_case=self))
        self.add_step(Analysis(test_case=self))

    def validate(self):
        """
        Validate variables against a baseline
        """
        compare_variables(test_case=self,
                          variables=['layerThickness', 'normalVelocity'],
                          filename1='forward/output.nc')
