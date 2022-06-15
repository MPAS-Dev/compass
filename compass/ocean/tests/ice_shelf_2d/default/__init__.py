from compass.testcase import TestCase
from compass.ocean.tests.ice_shelf_2d.initial_state import InitialState
from compass.ocean.tests.ice_shelf_2d.ssh_adjustment import SshAdjustment
from compass.ocean.tests.ice_shelf_2d.forward import Forward
from compass.ocean.tests.ice_shelf_2d.viz import Viz
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

    coord_type : str
        The type of vertical coordinate (``z-star``, ``z-level``, etc.)
    """

    def __init__(self, test_group, resolution, coord_type):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.ice_shelf_2d.IceShelf2d
            The test group that this test case belongs to

        resolution : str
            The resolution of the test case

        coord_type : str
            The type of vertical coordinate (``z-star``, ``z-level``, etc.)
        """
        name = 'default'
        self.resolution = resolution
        self.coord_type = coord_type
        subdir = '{}/{}/{}'.format(resolution, coord_type, name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(
            InitialState(test_case=self, resolution=resolution))
        self.add_step(
            SshAdjustment(test_case=self,  ntasks=4, openmp_threads=1))
        self.add_step(
            Forward(test_case=self, ntasks=4, openmp_threads=1,
                    resolution=resolution,  with_frazil=True))
        self.add_step(Viz(test_case=self), run_by_default=False)

    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        ice_shelf_2d.configure(self.resolution, self.coord_type, self.config)

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
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
