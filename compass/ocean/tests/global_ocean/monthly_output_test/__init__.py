from compass.ocean.tests.global_ocean.forward import (
    ForwardStep,
    ForwardTestCase,
)
from compass.validate import compare_variables


class MonthlyOutputTest(ForwardTestCase):
    """
    A test case to test the output for the TimeSeriesStatMonthly analysis
    member in E3SM.
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

        time_integrator : {'split_explicit_ab2', 'RK4'}
            The time integrator to use for the forward run
        """
        super().__init__(test_group=test_group, mesh=mesh, init=init,
                         time_integrator=time_integrator,
                         name='monthly_output_test')

        step = ForwardStep(test_case=self, mesh=mesh, init=init,
                           time_integrator=time_integrator, ntasks=4,
                           openmp_threads=1)

        module = self.__module__
        step.add_output_file(filename='output.nc')
        step.add_output_file(
            filename='analysis_members/'
                     'mpaso.hist.am.timeSeriesStatsMonthly.0001-01-01.nc')
        step.add_namelist_file(module, 'namelist.forward')
        if mesh.mesh_name in ['QU240', 'QUwISC240']:
            # a shorter time step is needed to prevent crashes
            step.add_namelist_options(dict(config_dt='00:30:00'))
        step.add_streams_file(module, 'streams.forward')
        self.add_step(step)

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = [
            'timeMonthly_avg_activeTracers_temperature',
            'timeMonthly_avg_activeTracers_salinity',
            'timeMonthly_avg_layerThickness', 'timeMonthly_avg_normalVelocity',
            'timeMonthly_avg_ssh']

        compare_variables(
            test_case=self, variables=variables,
            filename1='forward/analysis_members/'
                      'mpaso.hist.am.timeSeriesStatsMonthly.0001-01-01.nc')
