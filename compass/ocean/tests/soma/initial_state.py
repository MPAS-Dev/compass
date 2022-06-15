import xarray

from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull

from compass.step import Step
from compass.model import run_model


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for SOMA test cases

    Attributes
    ----------
    resolution : str
        The resolution of the test case
    """

    def __init__(self, test_case, resolution, with_surface_restoring,
                 three_layer):
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

        super().__init__(test_case=test_case, name='initial_state',
                         ntasks=res_params['cores'],
                         min_tasks=res_params['min_tasks'])

        mesh_filenames = {'32km': 'SOMA_32km_grid.161202.nc',
                          '16km': 'SOMA_16km_grid.161202.nc',
                          '8km': 'SOMA_8km_grid.161202.nc',
                          '4km': 'SOMA_4km_grid.161202.nc'}
        if resolution not in mesh_filenames:
            raise ValueError(f'Unexpected SOMA resolution: {resolution}')

        self.add_input_file(filename='base_mesh.nc',
                            target=mesh_filenames[resolution],
                            database='mesh_database')

        self.add_model_as_input()

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

        self.add_namelist_file(package, 'namelist.init', mode='init',
                               out_name='namelist_mark_land.ocean')

        mark_land_options = dict(
            config_write_cull_cell_mask='.true.',
            config_block_decomp_file_prefix="'base_graph.info.part.'",
            config_proc_decomp_file_prefix="'base_graph.info.part.'")

        mark_land_options.update(options)

        self.add_namelist_options(options=mark_land_options, mode='init',
                                  out_name='namelist_mark_land.ocean')

        self.add_streams_file(
            package, 'streams.init', mode='init',
            template_replacements={'mesh_filename': 'mesh.nc',
                                   'init_filename': 'masked_initial_state.nc',
                                   'forcing_filename': 'masked_forcing.nc'},
            out_name='streams_mark_land.ocean')

        self.add_namelist_file(package, 'namelist.init', mode='init',
                               out_name='namelist.ocean')
        options['config_write_cull_cell_mask'] = '.false.'
        self.add_namelist_options(options=options, mode='init',
                                  out_name='namelist.ocean')

        self.add_streams_file(
            package, 'streams.init', mode='init',
            template_replacements={'mesh_filename': 'culled_mesh.nc',
                                   'init_filename': 'initial_state.nc',
                                   'forcing_filename': 'forcing.nc'},
            out_name='streams.ocean')

        for file in ['initial_state.nc', 'forcing.nc', 'graph.info']:
            self.add_output_file(filename=file)

    def run(self):
        """
        Run this step of the test case
        """

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

        for out_name in ['namelist_mark_land.ocean', 'namelist.ocean']:
            self.update_namelist_at_runtime(options=options, out_name=out_name)
        ds_mesh = convert(xarray.open_dataset('base_mesh.nc'),
                          graphInfoFileName='base_graph.info',
                          logger=self.logger)
        write_netcdf(ds_mesh, 'mesh.nc')

        run_model(self, namelist='namelist_mark_land.ocean',
                  streams='streams_mark_land.ocean',
                  graph_file='base_graph.info')

        ds_mesh = cull(xarray.open_dataset('masked_initial_state.nc'),
                       graphInfoFileName='graph.info',
                       logger=self.logger)
        write_netcdf(ds_mesh, 'culled_mesh.nc')

        run_model(self, namelist='namelist.ocean', streams='streams.ocean',
                  graph_file='graph.info')
