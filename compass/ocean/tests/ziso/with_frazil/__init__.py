from compass.testcase import TestCase
from compass.ocean.tests.ziso.initial_state import InitialState
from compass.ocean.tests.ziso.forward import Forward
from compass.ocean.tests import ziso
from compass.validate import compare_variables


class WithFrazil(TestCase):
    """
    The with frazil test case for the ZISO test group simply creates the mesh
    and initial condition, then performs a short forward run including frazil
    formation.

    Attributes
    ----------
    resolution : str
        The resolution of the test case
    """

    def __init__(self, test_group, resolution):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.ziso.Ziso
            The test group that this test case belongs to

        resolution : str
            The resolution of the test case
        """
        name = 'with_frazil'
        self.resolution = resolution
        subdir = '{}/{}'.format(resolution, name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(
            InitialState(test_case=self, resolution=resolution,
                         with_frazil=True))
        self.add_step(
            Forward(test_case=self, resolution=resolution,
                    with_analysis=False, with_frazil=True))

    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        ziso.configure(self.name, self.resolution, self.config)

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['temperature', 'layerThickness']
        compare_variables(
            test_case=self, variables=variables,
            filename1='forward/output/output.0001-01-01_00.00.00.nc')

        variables = ['accumulatedFrazilIceMass',
                     'accumulatedFrazilIceSalinity',
                     'seaIceEnergy', 'frazilLayerThicknessTendency',
                     'frazilTemperatureTendency', 'frazilSalinityTendency',
                     'frazilSurfacePressure',
                     'accumulatedLandIceFrazilMass']
        compare_variables(test_case=self, variables=variables,
                          filename1='forward/frazil.nc')
