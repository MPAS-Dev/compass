from compass.validate import compare_variables
from compass.ocean.tests.global_ocean.forward import ForwardTestCase, \
    ForwardStep


class TimeSeriesStatsRestartTest(ForwardTestCase):
    """
    A test case to test bit-for-bit restart capabilities from the
    TimeSeriesStats analysis members in E3SM.

    Attributes
    ----------
    analysis : {'Daily', 'Monthly'}
        The suffix for the ``timeSeriesStats`` analysis member to check.

    """

    def __init__(self, test_group, mesh, init, analysis):
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

        analysis : {'Daily', 'Monthly'}
            The suffix for the ``timeSeriesStats`` analysis member to check.
        """
        name = f'{analysis.lower()}_restart_test'
        time_integrator = 'split_explicit'
        super().__init__(test_group=test_group, mesh=mesh, init=init,
                         time_integrator=time_integrator, name=name)
        module = self.__module__
        self.analysis = analysis

        output_interval = '0000-00-00_04:00:00'

        replacements = dict(analysis=analysis,
                            output_interval=output_interval)

        restart_filename = '../restarts/rst.0001-01-01_04.00.00.nc'
        analysis_restart_filename = \
            f'../restarts/rst.timeSeriesStats{analysis}.0001-01-01_04.00.00.nc'
        analysis_filename = \
            f'analysis_members/mpaso.hist.am.timeSeriesStats{analysis}.0001-01-01_08.00.00.nc'

        for part in ['full', 'restart']:
            name = f'{part}_run'
            step = ForwardStep(test_case=self, mesh=mesh, init=init,
                               time_integrator=time_integrator, name=name,
                               subdir=name, ntasks=4, openmp_threads=1)

            step.add_namelist_file(module, f'namelist.{part}')
            step.add_streams_file(module, 'streams.forward',
                                  template_replacements=replacements)
            if part == 'full':
                step.add_output_file(restart_filename)
                step.add_output_file(analysis_restart_filename)
            else:
                step.add_input_file(restart_filename)
                step.add_input_file(analysis_restart_filename)
            step.add_output_file(analysis_filename)
            self.add_step(step)

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        analysis = self.analysis
        variables = [
            'Time', 'Time_bnds',
            f'time{analysis}_avg_activeTracers_temperature',
            f'time{analysis}_avg_activeTracers_salinity',
            f'time{analysis}_avg_layerThickness',
            f'time{analysis}_avg_normalVelocity',
            f'time{analysis}_avg_ssh']

        analysis_filename = \
            f'analysis_members/mpaso.hist.am.timeSeriesStats{analysis}.0001-02-01.nc'

        compare_variables(
            test_case=self, variables=variables,
            filename1=f'full_run/{analysis_filename}',
            filename2=f'restart_run/{analysis_filename}')
