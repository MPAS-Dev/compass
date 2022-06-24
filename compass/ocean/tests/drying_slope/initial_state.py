from compass.model import ModelStep


class InitialState(ModelStep):
    """
    A step for creating an initial condition for drying slope test cases
    """
    def __init__(self, test_case, coord_type):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.drying_slope.default.Default
            The test case this step belongs to

        coord_type : {'sigma', 'single_layer'}
            The type of vertical coordinate
        """
        super().__init__(test_case=test_case, name='initial_state', ntasks=1,
                         min_tasks=1, openmp_threads=1)

        self.coord_type = coord_type

        self.add_namelist_file('compass.ocean.tests.drying_slope',
                               'namelist.init', mode='init')

        self.add_streams_file('compass.ocean.tests.drying_slope',
                              'streams.init', mode='init')

        self.add_input_file(filename='culled_mesh.nc',
                            target=f'../mesh/culled_mesh.nc')

        self.add_input_file(filename='graph.info',
                            target=f'../mesh/culled_graph.info')

        self.add_output_file('ocean.nc')
        self.add_output_file('init_mode_forcing_data.nc')

    def runtime_setup(self):
        """
        Set the number of layers based on the coordinate type
        """
        super().runtime_setup()
        section = self.config['vertical_grid']
        coord_type = self.coord_type
        if coord_type == 'single_layer':
            options = {'config_tidal_boundary_vert_levels': '1'}
            self.update_namelist_at_runtime(options)
        else:
            vert_levels = section.get('vert_levels')
            options = {'config_tidal_boundary_vert_levels': f'{vert_levels}',
                       'config_tidal_boundary_layer_type': f"'{coord_type}'"}
            self.update_namelist_at_runtime(options)
