from compass.config import CompassConfigParser
from compass.landice.tests.mismipplus.setup_mesh import SetupMesh
from compass.testcase import TestCase


class SpinUp(TestCase):
    """
    Test case for create the MISMIP+ mesh, initial conditions,
    input files, and runs a short segment of the spin up experiments
    """
    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.test.mismipplus
            The test group that this test case belongs to

        """
        name = "spin_up"

        super().__init__(test_group=test_group, name=name)

        config = CompassConfigParser()
        module = 'compass.landice.tests.mismipplus.spin_up'
        # add from config
        config.add_from_package(module, 'mesh_gen.cfg')
        resolution = config.getint('mesh', 'resolution')

        # Setting up steps of test case
        self.add_step(SetupMesh(test_case=self, resolution=resolution))
