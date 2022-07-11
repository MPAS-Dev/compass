from compass.model import ModelStep


class InitialState(ModelStep):
    """
    A step for creating a mesh and initial condition, either with or without
    a mask for culling land cells, for SOMA test cases

    Attributes
    ----------
    resolution : str
        The resolution of the test case
    """

    def __init__(self, test_case, resolution, with_surface_restoring,
                 three_layer, mark_land):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : str
            The resolution of the test case

        with_surface_restoring : bool
            Whether surface restoring is included in the simulation

        three_layer : bool
            Whether to use only 3 vertical layers and no continental shelf

        mark_land : bool
            Whether to mark land cells for culling
        """
        self.resolution = resolution

        res_params = {'32km': {'cores': 4,
                               'min_tasks': 1},
                      '16km': {'cores': 10,
                               'min_tasks': 1},
                      '8km': {'cores': 40,
                              'min_tasks': 1},
                      '4km': {'cores': 160,
                              'min_tasks': 1}}

        if resolution not in res_params:
            raise ValueError(
                f'Unsupported resolution {resolution}. Supported values are: '
                f'{list(res_params)}')

        res_params = res_params[resolution]

        if mark_land:
            name = 'init_on_base_mesh'
        else:
            name = 'initial_state'

        super().__init__(test_case=test_case, name=name,
                         ntasks=res_params['cores'],
                         min_tasks=res_params['min_tasks'],
                         openmp_threads=1)

        package = 'compass.ocean.tests.soma'

        options = dict()
        if with_surface_restoring:
            options['config_soma_use_surface_temp_restoring'] = '.true.'
            options['config_use_activeTracers_surface_restoring'] = '.true.'

        if three_layer:
            options['config_soma_vert_levels'] = '3'
            options['config_vertical_grid'] = "'uniform'"
        else:
            options['config_soma_vert_levels'] = '60'
            options['config_vertical_grid'] = "'60layerPHC'"

        if mark_land:
            options['config_write_cull_cell_mask'] = '.true.'
        else:
            options['config_write_cull_cell_mask'] = '.false.'

        self.add_namelist_file(package, 'namelist.init', mode='init',
                               out_name='namelist.ocean')

        self.add_namelist_options(options=options, mode='init',
                                  out_name='namelist.ocean')

        self.add_streams_file(
            package, 'streams.init', mode='init',
            out_name='streams.ocean')

        if mark_land:
            self.add_input_file(filename='mesh.nc',
                                target='../base_mesh/mesh.nc')
            self.add_input_file(filename='graph.info',
                                target='../base_mesh/base_graph.info')
        else:
            self.add_input_file(filename='mesh.nc',
                                target='../culled_mesh/culled_mesh.nc')
            self.add_input_file(filename='graph.info',
                                target='../culled_mesh/culled_graph.info')

        for file in ['initial_state.nc', 'forcing.nc']:
            self.add_output_file(filename=file)

    def runtime_setup(self):
        """
        Set namelist options from config options
        """
        super().runtime_setup()

        config = self.config
        section = config['soma']
        options = dict(
            config_eos_linear_alpha=section.get('eos_linear_alpha'),
            config_soma_density_difference=section.get('density_difference'),
            config_soma_surface_temperature=section.get('surface_temperature'),
            config_soma_surface_salinity=section.get('surface_salinity'),
            config_soma_salinity_gradient=section.get('salinity_gradient'),
            config_soma_thermocline_depth=section.get('thermocline_depth'),
            config_soma_density_difference_linear=section.get(
                'density_difference_linear'),
            config_soma_phi=section.get('phi'),
            config_soma_shelf_depth=section.get('shelf_depth'),
            config_soma_bottom_depth=section.get('bottom_depth'))

        self.update_namelist_at_runtime(options=options,
                                        out_name='namelist.ocean')
