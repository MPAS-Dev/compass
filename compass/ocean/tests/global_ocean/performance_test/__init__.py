from compass.validate import compare_variables, compare_timers
from compass.ocean.tests.global_ocean.forward import ForwardTestCase, \
    ForwardStep


class PerformanceTest(ForwardTestCase):
    """
    A test case for performing a short forward run with an MPAS-Ocean global
    initial condition assess performance and compare with previous results
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
        super().__init__(test_group=test_group, mesh=mesh, init=init,
                         time_integrator=time_integrator,
                         name='performance_test')

        step = ForwardStep(test_case=self, mesh=mesh, init=init,
                           time_integrator=time_integrator)

        step.add_output_file(filename='output.nc')
        if mesh.with_ice_shelf_cavities:
            module = self.__module__
            step.add_namelist_file(module, 'namelist.wisc')
            step.add_streams_file(module, 'streams.wisc')
            step.add_output_file(filename='land_ice_fluxes.nc')
        self.add_step(step)

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['temperature', 'salinity', 'layerThickness',
                     'normalVelocity']
        if self.init.with_bgc:
            variables.extend(
                ['PO4', 'NO3', 'SiO3', 'NH4', 'Fe', 'O2', 'DIC', 'DIC_ALT_CO2',
                 'ALK', 'DOC', 'DON', 'DOFe', 'DOP', 'DOPr', 'DONr', 'zooC',
                 'spChl', 'spC', 'spFe', 'spCaCO3', 'diatChl', 'diatC',
                 'diatFe', 'diatSi', 'diazChl', 'diazC', 'diazFe', 'phaeoChl',
                 'phaeoC', 'phaeoFe'])

        compare_variables(test_case=self, variables=variables,
                          filename1='forward/output.nc')

        if self.mesh.with_ice_shelf_cavities:
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

        timers = ['time integration']
        compare_timers(timers, self.config, self.work_dir, rundir1='forward')
