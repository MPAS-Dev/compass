from compass.ocean.tests.global_ocean.data_ice_shelf_melt.remap_ice_shelf_melt import (  # noqa: E501
    RemapIceShelfMelt,
)
from compass.ocean.tests.global_ocean.forward import (
    ForwardStep,
    get_forward_subdir,
)
from compass.testcase import TestCase
from compass.validate import compare_variables


class DataIceShelfMelt(TestCase):
    """
    A test case for remapping observed melt rates to the MPAS grid and then
    performing a short forward run with these data (prescribed) melt rates

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.Mesh
        The test case that produces the mesh for this run

    init : compass.ocean.tests.global_ocean.init.Init
        The test case that produces the initial condition for this run
    """

    def __init__(self, test_group, mesh, init, time_integrator):
        """
        Create test case

        Parameters
        ----------
        test_group : compass.ocean.tests.global_ocean.GlobalOcean
            The global ocean test group that this test case belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that produces the mesh for this run

        init : compass.ocean.tests.global_ocean.init.Init
            The test case that produces the initial condition for this run

        time_integrator : {'split_explicit', 'RK4'}
            The time integrator to use for the forward run
        """
        self.mesh = mesh
        self.init = init
        name = 'data_ice_shelf_melt'
        subdir = get_forward_subdir(init.init_subdir, time_integrator, name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.add_step(RemapIceShelfMelt(test_case=self, mesh=mesh))

        step = ForwardStep(test_case=self, mesh=mesh, init=init,
                           time_integrator=time_integrator)

        module = self.__module__
        step.add_input_file(
            filename='prescribed_ismf_adusumilli2020.nc',
            target='../remap_ice_shelf_melt/prescribed_ismf_adusumilli2020.nc')
        step.add_namelist_file(module, 'namelist.forward')
        step.add_streams_file(module, 'streams.forward')
        step.add_output_file(filename='land_ice_fluxes.nc')
        step.add_output_file(filename='output.nc')
        self.add_step(step)

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        self.init.configure(config=self.config)

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['temperature', 'salinity', 'layerThickness',
                     'normalVelocity']

        compare_variables(test_case=self, variables=variables,
                          filename1='forward/output.nc')

        variables = [
            'ssh', 'landIcePressure', 'landIceDraft', 'landIceFraction',
            'landIceMask', 'landIceFrictionVelocity', 'topDrag',
            'topDragMagnitude', 'landIceFreshwaterFlux', 'landIceHeatFlux',
            'heatFluxToLandIce', 'landIceBoundaryLayerTemperature',
            'landIceBoundaryLayerSalinity', 'landIceHeatTransferVelocity',
            'landIceSaltTransferVelocity', 'landIceInterfaceTemperature',
            'landIceInterfaceSalinity', 'accumulatedLandIceMass',
            'accumulatedLandIceHeat']

        compare_variables(test_case=self, variables=variables,
                          filename1='forward/land_ice_fluxes.nc')
