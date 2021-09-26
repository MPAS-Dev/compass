from compass.ocean.tests.global_ocean.dynamic_adjustment import \
    DynamicAdjustment
from compass.ocean.tests.global_ocean.forward import ForwardStep
from compass.ocean.tests.global_ocean.mesh.qu import get_qu_dts, set_qu_cores


class QUDynamicAdjustment(DynamicAdjustment):
    """
    A test case performing dynamic adjustment (dissipating fast-moving waves)
    from an initial condition on the WC14 MPAS-Ocean mesh

    Attributes
    ----------
    restart_filenames : list of str
        A list of restart files from each dynamic-adjustment step
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
        if time_integrator != 'split_explicit':
            raise ValueError('{} dynamic adjustment not defined for {}'.format(
                mesh.mesh_name, time_integrator))

        restart_times = ['0001-01-01_06:00:00']
        restart_filenames = [
            'restarts/rst.{}.nc'.format(restart_time.replace(':', '.'))
            for restart_time in restart_times]

        super().__init__(test_group=test_group, mesh=mesh, init=init,
                         time_integrator=time_integrator,
                         restart_filenames=restart_filenames)

        module = self.__module__

        global_stats = {'config_AM_globalStats_enable': '.true.',
                        'config_AM_globalStats_compute_on_startup': '.true.',
                        'config_AM_globalStats_write_on_startup': '.true.'}

        # fist and final step
        step_name = 'simulation'
        step = ForwardStep(test_case=self, mesh=mesh, init=init,
                           time_integrator=time_integrator, name=step_name,
                           subdir=step_name)

        namelist_options = {
            'config_run_duration': "'00-00-00_06:00:00'"}
        namelist_options.update(global_stats)
        step.add_namelist_options(namelist_options)

        stream_replacements = {
            'output_interval': '00-00-00_06:00:00',
            'restart_interval': '00-00-00_06:00:00'}
        step.add_streams_file(module, 'streams.template',
                              template_replacements=stream_replacements)

        step.add_output_file(filename='../{}'.format(restart_filenames[0]))
        step.add_output_file(filename='output.nc')
        self.add_step(step)

        self.restart_filenames = restart_filenames

    def configure(self):
        """ Update the number of cores and min_cores the forward step """

        super().configure()
        set_qu_cores(self.config)
        dt, btr_dt = get_qu_dts(self.config)
        for step in self.steps.values():
            step.add_namelist_options({'config_dt': dt,
                                       'config_btr_dt': btr_dt})

    def run(self):
        dt, btr_dt = get_qu_dts(self.config)
        for step in self.steps.values():
            step.update_namelist_at_runtime(options={'config_dt': dt,
                                                     'config_btr_dt': btr_dt},
                                            out_name='namelist.ocean')
        super().run()
