from importlib.resources import contents

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
                         time_integrator='split_explicit_ab2',
                         name='ssh_adjustment')

        self.add_namelist_options({'config_AM_globalStats_enable': '.false.'})
        self.add_namelist_file('compass.ocean.namelists',
                               'namelist.ssh_adjust')

        self.add_streams_file('compass.ocean.streams', 'streams.ssh_adjust')
        self.add_streams_file('compass.ocean.tests.global_ocean.init',
                              'streams.ssh_adjust')

        mesh_package = test_case.mesh.package
        mesh_package_contents = list(contents(mesh_package))
        mesh_namelist = 'namelist.ssh_adjust'
        if mesh_namelist in mesh_package_contents:
            self.add_namelist_file(mesh_package, mesh_namelist)

        mesh_streams = 'streams.ssh_adjust'
        if mesh_streams in mesh_package_contents:
            self.add_streams_file(mesh_package, mesh_streams)

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
        update_pio = config.getboolean('global_ocean', 'forward_update_pio')
        convert_to_cdf5 = config.getboolean('ssh_adjustment',
                                            'convert_to_cdf5')
        adjust_ssh(variable='landIcePressure', iteration_count=iteration_count,
                   step=self, update_pio=update_pio,
                   convert_to_cdf5=convert_to_cdf5)
