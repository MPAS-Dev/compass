from compass.ocean.tests.global_ocean.dynamic_adjustment import \
    DynamicAdjustment
from compass.ocean.tests.global_ocean.forward import ForwardStep


class QU240DynamicAdjustment(DynamicAdjustment):
    """
    A test case performing dynamic adjustment (dissipating fast-moving waves)
    from an initial condition on the QU240 MPAS-Ocean mesh
    """

    def __init__(self, test_group, mesh, init, time_integrator):
        """
        Create the test case

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
        restart_times = ['0001-01-02_00:00:00', '0001-01-03_00:00:00']
        restart_filenames = [
            'restarts/rst.{}.nc'.format(restart_time.replace(':', '.'))
            for restart_time in restart_times]

        super().__init__(test_group=test_group, mesh=mesh, init=init,
                         time_integrator=time_integrator,
                         restart_filenames=restart_filenames)

        module = self.__module__

        # first step
        step_name = 'damped_adjustment_1'
        step = ForwardStep(test_case=self, mesh=mesh, init=init,
                           time_integrator=time_integrator, name=step_name,
                           subdir=step_name)

        namelist_options = {
            'config_run_duration': "'00-00-01_00:00:00'",
            'config_Rayleigh_friction': '.true.',
            'config_Rayleigh_damping_coeff': '1.0e-4'}
        step.add_namelist_options(namelist_options)

        stream_replacements = {
            'output_interval': '00-00-01_00:00:00',
            'restart_interval': '00-00-01_00:00:00'}
        step.add_streams_file(module, 'streams.template',
                              template_replacements=stream_replacements)

        step.add_output_file(filename='../{}'.format(restart_filenames[0]))
        self.add_step(step)

        # final step
        step_name = 'simulation'
        step = ForwardStep(test_case=self, mesh=mesh, init=init,
                           time_integrator=time_integrator, name=step_name,
                           subdir=step_name)

        namelist_options = {
            'config_run_duration': "'00-00-01_00:00:00'",
            'config_do_restart': '.true.',
            'config_start_time': "'{}'".format(restart_times[0])}
        step.add_namelist_options(namelist_options)

        stream_replacements = {
            'output_interval': '00-00-01_00:00:00',
            'restart_interval': '00-00-01_00:00:00'}
        step.add_streams_file(module, 'streams.template',
                              template_replacements=stream_replacements)

        step.add_input_file(filename='../{}'.format(restart_filenames[0]))
        step.add_output_file(filename='../{}'.format(restart_filenames[1]))
        step.add_output_file(filename='output.nc')
        self.add_step(step)
