from compass.ocean.iceshelf import adjust_ssh
from compass.ocean.tests.global_ocean.forward import ForwardStep


class SshAdjustment(ForwardStep):
    """
    A step for iteratively adjusting the pressure from the weight of the ice
    shelf to match the sea-surface height as part of ice-shelf 2D test cases
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.init.Init
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, mesh=test_case.mesh,
                         time_integrator='split_explicit',
                         name='ssh_adjustment')

        self.add_namelist_options({'config_AM_globalStats_enable': '.false.'})
        self.add_namelist_file('compass.ocean.namelists',
                               'namelist.ssh_adjust')

        self.add_streams_file('compass.ocean.streams', 'streams.ssh_adjust')
        self.add_streams_file('compass.ocean.tests.global_ocean.init',
                              'streams.ssh_adjust')

        mesh_path = test_case.mesh.get_cull_mesh_path()
        init_path = test_case.steps['initial_state'].path

        self.add_input_file(
            filename='adjusting_init0.nc',
            work_dir_target=f'{init_path}/initial_state.nc')
        self.add_input_file(
            filename='forcing_data.nc',
            work_dir_target=f'{init_path}/init_mode_forcing_data.nc')
        self.add_input_file(
            filename='graph.info',
            work_dir_target=f'{mesh_path}/culled_graph.info')

        self.add_output_file(filename='adjusted_init.nc')

    def run(self):
        """
        Run this step of the testcase
        """
        config = self.config
        iteration_count = config.getint('ssh_adjustment', 'iterations')
        adjust_ssh(variable='landIcePressure', iteration_count=iteration_count,
                   step=self)
