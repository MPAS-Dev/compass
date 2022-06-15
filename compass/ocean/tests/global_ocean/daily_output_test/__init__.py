from compass.validate import compare_variables
from compass.ocean.tests.global_ocean.forward import ForwardTestCase, \
    ForwardStep


class DailyOutputTest(ForwardTestCase):
    """
    A test case to test the output for the TimeSeriesStatMonthly analysis
    member in E3SM.  In this test, the analysis member for daily output is used
    instead for efficiency.
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
                         name='daily_output_test')

        step = ForwardStep(test_case=self, mesh=mesh, init=init,
                           time_integrator=time_integrator, ntasks=4,
                           openmp_threads=1)

        module = self.__module__
        step.add_output_file(filename='output.nc')
        step.add_output_file(
            filename='analysis_members/'
                     'mpaso.hist.am.timeSeriesStatsDaily.0001-01-01.nc')
        step.add_namelist_file(module, 'namelist.forward')
        step.add_streams_file(module, 'streams.forward')
        self.add_step(step)

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = [
            'timeDaily_avg_activeTracers_temperature',
            'timeDaily_avg_activeTracers_salinity',
            'timeDaily_avg_layerThickness', 'timeDaily_avg_normalVelocity',
            'timeDaily_avg_ssh']

        compare_variables(
            test_case=self, variables=variables,
            filename1='forward/analysis_members/'
                      'mpaso.hist.am.timeSeriesStatsDaily.0001-01-01.nc')
