from compass.testcase import TestCase
from compass.ocean.tests.ice_shelf_2d.initial_state import InitialState
from compass.ocean.tests.ice_shelf_2d.ssh_adjustment import SshAdjustment
from compass.ocean.tests.ice_shelf_2d.forward import Forward
from compass.ocean.tests import ice_shelf_2d
from compass.validate import compare_variables


class Default(TestCase):
    """
    The default ice-shelf 2D test case, which performs a short forward run with
    the z-star vertical coordinate and with 15 iterations of adjustment to make
    the pressure from the weight of the ice shelf match the sea-surface height

    Attributes
    ----------
    resolution : str
        The horizontal resolution of the test case
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
        name = 'default'
        self.resolution = resolution
        subdir = '{}/{}'.format(resolution, name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(
            InitialState(test_case=self, resolution=resolution))
        self.add_step(
            SshAdjustment(test_case=self,  cores=4, threads=1))
        self.add_step(
            Forward(test_case=self, cores=4, threads=1, resolution=resolution,
                    with_frazil=True))

    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        ice_shelf_2d.configure(self.name, self.resolution, self.config)

    def run(self):
        """
        Run each step of the test case
        """
        # run the steps
        super().run()

        # perform validation
        variables = ['temperature', 'salinity', 'layerThickness',
                     'normalVelocity']
        compare_variables(test_case=self, variables=variables,
                          filename1='forward/output.nc')

        variables = \
            ['ssh', 'landIcePressure', 'landIceDraft', 'landIceFraction',
             'landIceMask', 'landIceFrictionVelocity', 'topDrag',
             'topDragMagnitude', 'landIceFreshwaterFlux',
             'landIceHeatFlux', 'heatFluxToLandIce',
             'landIceBoundaryLayerTemperature', 'landIceBoundaryLayerSalinity',
             'landIceHeatTransferVelocity', 'landIceSaltTransferVelocity',
             'landIceInterfaceTemperature', 'landIceInterfaceSalinity',
             'accumulatedLandIceMass', 'accumulatedLandIceHeat']
        compare_variables(test_case=self, variables=variables,
                          filename1='forward/land_ice_fluxes.nc')
