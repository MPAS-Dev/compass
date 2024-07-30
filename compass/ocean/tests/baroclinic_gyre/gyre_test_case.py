from compass.mesh import QuasiUniformSphericalMeshStep
from compass.ocean.tests.baroclinic_gyre.cull_mesh import CullMesh
from compass.ocean.tests.baroclinic_gyre.forward import Forward
from compass.ocean.tests.baroclinic_gyre.initial_state import InitialState
from compass.ocean.tests.baroclinic_gyre.moc import Moc
from compass.testcase import TestCase
from compass.validate import compare_variables


class GyreTestCase(TestCase):
    """
    A class to define the baroclinic gyre test cases

    Attributes
    ----------
    resolution : float
        The resolution of the test case (m)
    """

    def __init__(self, test_group, resolution, long):
        """
        Create the test case

        Parameters
        ----------
        test_group :
        compass.ocean.tests.baroclinic_gyre.BaroclinicGyre
            The test group that this test case belongs to

        resolution : float
            The resolution of the test case (m)

        long : bool
            Whether to run a long (3-year) simulation to quasi-equilibrium
        """
        name = 'performance_test'
        self.resolution = resolution
        self.long = long

        if long:
            name = '3_year_test'

        if resolution >= 1e3:
            res_name = f'{int(resolution/1e3)}km'
        else:
            res_name = f'{int(resolution)}m'
        subdir = f'{res_name}/{name}'
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(QuasiUniformSphericalMeshStep(
            test_case=self, cell_width=int(resolution / 1e3)))
        self.add_step(CullMesh(test_case=self))
        self.add_step(
            InitialState(test_case=self, resolution=resolution))
        self.add_step(
            Forward(test_case=self, resolution=resolution,
                    long=long))
        if long:
            self.add_step(
                Moc(test_case=self, resolution=resolution))

    def configure(self):
        """
        Set config options for the test case
        """
        config = self.config
        config.add_from_package('compass.mesh', 'mesh.cfg')

    def validate(self):
        """
        Validate variables against a baseline
        """
        compare_variables(test_case=self,
                          variables=['layerThickness', 'temperature',
                                     'ssh'],
                          filename1='forward/output/'
                                    'output.0001-01-01_00.00.00.nc')
