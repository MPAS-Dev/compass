from importlib.resources import path

from compass.io import symlink
from compass.config import add_config
from compass.landice.tests.enthalpy_benchmark.setup_mesh import SetupMesh
from compass.landice.tests.enthalpy_benchmark.run_model import RunModel
from compass.landice.tests.enthalpy_benchmark.A.visualize import Visualize
from compass.testcase import TestCase


class B(TestCase):
    """
    The Kleiner enthalpy benchmark test case B

    Attributes
    ----------
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.enthalpy_benchmark.EnthalpyBenchmark
            The test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='B')

        SetupMesh(test_case=self)
        RunModel(test_case=self, name='run_model', cores=1, threads=1)
        Visualize(test_case=self)

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        add_config(self.config, 'compass.landice.tests.enthalpy_benchmark.B',
                   'B.cfg', exception=True)

        with path('compass.landice.tests.enthalpy_benchmark', 'README') as \
                target:
            symlink(str(target), '{}/README'.format(self.work_dir))

    # no run() method needed: we just run the steps, the default behavior
