from compass.testcase import TestCase
from compass.ocean.tests.ice_shelf_2d.initial_state import InitialState
from compass.ocean.tests.ice_shelf_2d.ssh_adjustment import SshAdjustment
from compass.ocean.tests.ice_shelf_2d.forward import Forward
from compass.ocean.tests import ice_shelf_2d
from compass.validate import compare_variables


class RestartTest(TestCase):
    """
    A restart test case for the ice-shelf 2D test case test group, which makes
    sure the model produces identical results with one longer run and two
    shorter runs with a restart in between.

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
        test_group : compass.ocean.tests.ice_shelf_2d.IceShelf2d
            The test group that this test case belongs to

        resolution : str
            The resolution of the test case
        """
        name = 'restart_test'
        self.resolution = resolution
        subdir = '{}/{}'.format(resolution, name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(
            InitialState(test_case=self, resolution=resolution))
        self.add_step(
            SshAdjustment(test_case=self, cores=4, threads=1))

        for part in ['full', 'restart']:
            name = '{}_run'.format(part)
            step = Forward(test_case=self, name=name, subdir=name, cores=4,
                           threads=1, resolution=resolution, with_frazil=True)

            step.add_namelist_file(
                'compass.ocean.tests.ice_shelf_2d.restart_test',
                'namelist.{}'.format(part))
            step.add_streams_file(
                'compass.ocean.tests.ice_shelf_2d.restart_test',
                'streams.{}'.format(part))
            self.add_step(step)

    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        ice_shelf_2d.configure(self.name, self.resolution, self.config)

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        steps = self.steps_to_run
        if 'full_run' in steps and 'restart_run' in steps:
            variables = ['temperature', 'salinity', 'layerThickness',
                         'normalVelocity']
            compare_variables(test_case=self, variables=variables,
                              filename1='full_run/output.nc',
                              filename2='restart_run/output.nc')

            variables = ['ssh', 'landIcePressure', 'landIceDraft',
                         'landIceFraction',
                         'landIceMask', 'landIceFrictionVelocity', 'topDrag',
                         'topDragMagnitude', 'landIceFreshwaterFlux',
                         'landIceHeatFlux', 'heatFluxToLandIce',
                         'landIceBoundaryLayerTemperature',
                         'landIceBoundaryLayerSalinity',
                         'landIceHeatTransferVelocity',
                         'landIceSaltTransferVelocity',
                         'landIceInterfaceTemperature',
                         'landIceInterfaceSalinity', 'accumulatedLandIceMass',
                         'accumulatedLandIceHeat']
            compare_variables(test_case=self, variables=variables,
                              filename1='full_run/land_ice_fluxes.nc',
                              filename2='restart_run/land_ice_fluxes.nc')

            variables = ['accumulatedFrazilIceMass',
                         'accumulatedFrazilIceSalinity',
                         'seaIceEnergy', 'frazilLayerThicknessTendency',
                         'frazilTemperatureTendency', 'frazilSalinityTendency',
                         'frazilSurfacePressure',
                         'accumulatedLandIceFrazilMass']
            compare_variables(test_case=self, variables=variables,
                              filename1='full_run/frazil.nc',
                              filename2='restart_run/frazil.nc')
