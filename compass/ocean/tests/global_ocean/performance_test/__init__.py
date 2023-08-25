import os

from compass.ocean.inactive_top_cells import remove_inactive_top_cells_output
from compass.ocean.tests.global_ocean.forward import (
    ForwardStep,
    ForwardTestCase,
)
from compass.validate import compare_timers, compare_variables


class PerformanceTest(ForwardTestCase):
    """
    A test case for performing one or more short forward run with an MPAS-Ocean
    global initial condition assess performance, test prognostic and data
    melting (if ice-shelf cavities are present), and compare with previous
    results
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
        name = 'performance_test'
        if init.with_inactive_top_cells:
            if time_integrator == 'RK4':
                self.inactive_top_comp_subdir = os.path.join(
                    mesh.mesh_name, init.initial_condition, time_integrator,
                    name)
            else:
                self.inactive_top_comp_subdir = os.path.join(
                    mesh.mesh_name, init.initial_condition, name)
            name = f'{name}_inactive_top'
        super().__init__(test_group=test_group, mesh=mesh, init=init,
                         time_integrator=time_integrator, name=name)

        if mesh.with_ice_shelf_cavities:
            this_module = self.__module__
            # prognostic step
            flux_modes = dict(prognostic_ice_shelf_melt='standalone',
                              data_ice_shelf_melt='data')
            for step_name, flux_mode in flux_modes.items():
                step = ForwardStep(test_case=self, mesh=mesh, init=init,
                                   time_integrator=time_integrator,
                                   name=step_name,
                                   land_ice_flux_mode=flux_mode)
                step.add_streams_file(this_module, 'streams.wisc')
                step.add_output_file(filename='land_ice_fluxes.nc')
                step.add_output_file(filename='output.nc')
                self.add_step(step)
        else:
            step = ForwardStep(test_case=self, mesh=mesh, init=init,
                               time_integrator=time_integrator)

            step.add_output_file(filename='output.nc')
            self.add_step(step)

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        for step in self.steps.values():
            step_subdir = step.subdir
            variables = ['temperature', 'salinity', 'layerThickness',
                         'normalVelocity']

            compare_variables(test_case=self, variables=variables,
                              filename1=f'{step_subdir}/output.nc')

            if self.init.with_inactive_top_cells:
                filename2 = os.path.join(self.base_work_dir,
                                         self.mpas_core.name,
                                         self.test_group.name,
                                         self.inactive_top_comp_subdir,
                                         'forward/output.nc')
                if os.path.exists(filename2):
                    compare_variables(
                        test_case=self, variables=variables,
                        filename1=f'{step_subdir}/output_crop.nc',
                        filename2=filename2,
                        quiet=False, check_outputs=False,
                        skip_if_step_not_run=False)
                else:
                    self.logger.warn('The version of "performance_test" '
                                     'without inactive top cells was not run. '
                                     'Skipping validation.')

            if self.mesh.with_ice_shelf_cavities:
                variables = [
                    'ssh', 'landIcePressure', 'landIceDraft',
                    'landIceFraction', 'landIceMask',
                    'landIceFrictionVelocity', 'topDrag',
                    'topDragMagnitude', 'landIceFreshwaterFlux',
                    'landIceHeatFlux', 'heatFluxToLandIce',
                    'landIceBoundaryLayerTemperature',
                    'landIceBoundaryLayerSalinity',
                    'landIceHeatTransferVelocity',
                    'landIceSaltTransferVelocity',
                    'landIceInterfaceTemperature',
                    'landIceInterfaceSalinity', 'accumulatedLandIceMass',
                    'accumulatedLandIceHeat']

                compare_variables(
                    test_case=self, variables=variables,
                    filename1=f'{step_subdir}/land_ice_fluxes.nc')

            timers = ['time integration']
            compare_timers(self, timers, rundir1=step_subdir)
