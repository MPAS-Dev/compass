import os
from importlib.resources import contents

import xarray
from mpas_tools.io import write_netcdf

from compass.model import run_model
from compass.ocean.inactive_top_cells import remove_inactive_top_cells_output
from compass.ocean.plot import plot_initial_state, plot_vertical_grid
from compass.ocean.tests.global_ocean.metadata import (
    add_mesh_and_init_metadata,
)
from compass.ocean.vertical.grid_1d import generate_1d_grid, write_1d_grid
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for baroclinic channel
    test cases

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.mesh.MeshStep
        The step for creating the mesh

    initial_condition : {'WOA23', 'PHC', 'EN4_1900'}
        The initial condition dataset to use
    """
    def __init__(self, test_case, mesh, initial_condition,
                 with_inactive_top_cells):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.init.Init
            The test case this step belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that creates the mesh used by this test case

        initial_condition : {'WOA23', 'PHC', 'EN4_1900'}
            The initial condition dataset to use
        """
        if initial_condition not in ['WOA23', 'PHC', 'EN4_1900']:
            raise ValueError(f'Unknown initial_condition {initial_condition}')

        super().__init__(test_case=test_case, name='initial_state')
        self.mesh = mesh
        self.initial_condition = initial_condition
        self.with_inactive_top_cells = with_inactive_top_cells

        package = 'compass.ocean.tests.global_ocean.init'

        # generate the namelist, replacing a few default options
        self.add_namelist_file(package, 'namelist.init', mode='init')
        self.add_namelist_file(
            package, f'namelist.{initial_condition.lower()}',
            mode='init')
        if mesh.with_ice_shelf_cavities:
            self.add_namelist_file(package, 'namelist.wisc', mode='init')

        # generate the streams file
        self.add_streams_file(package, 'streams.init', mode='init')

        if mesh.with_ice_shelf_cavities:
            self.add_streams_file(package, 'streams.wisc', mode='init')

        mesh_package = mesh.package
        mesh_package_contents = list(contents(mesh_package))
        mesh_namelist = 'namelist.init'
        if mesh_namelist in mesh_package_contents:
            self.add_namelist_file(mesh_package, mesh_namelist, mode='init')

        mesh_streams = 'streams.init'
        if mesh_streams in mesh_package_contents:
            self.add_streams_file(mesh_package, mesh_streams, mode='init')

        options = {
            'config_global_ocean_topography_source': "'mpas_variable'"
        }
        self.add_namelist_options(options, mode='init')
        self.add_streams_file(package, 'streams.topo', mode='init')

        cull_step = self.mesh.steps['cull_mesh']
        target = os.path.join(cull_step.path, 'topography_culled.nc')
        self.add_input_file(filename='topography.nc',
                            work_dir_target=target)

        self.add_input_file(
            filename='wind_stress.nc',
            target='windStress.ncep_1958-2000avg.interp3600x2431.151106.nc',
            database='initial_condition_database')

        if initial_condition == 'WOA23':
            self.add_input_file(
                filename='woa23.nc',
                target='woa23_decav_0.25_extrap.20230416.nc',
                database='initial_condition_database')
        elif initial_condition == 'PHC':
            self.add_input_file(
                filename='temperature.nc',
                target='PotentialTemperature.01.filled.60levels.PHC.151106.nc',
                database='initial_condition_database')
            self.add_input_file(
                filename='salinity.nc',
                target='Salinity.01.filled.60levels.PHC.151106.nc',
                database='initial_condition_database')
        else:
            # EN4_1900
            self.add_input_file(
                filename='temperature.nc',
                target='PotentialTemperature.100levels.Levitus.'
                       'EN4_1900estimate.200813.nc',
                database='initial_condition_database')
            self.add_input_file(
                filename='salinity.nc',
                target='Salinity.100levels.Levitus.EN4_1900estimate.200813.nc',
                database='initial_condition_database')

        mesh_path = self.mesh.get_cull_mesh_path()

        self.add_input_file(
            filename='mesh.nc',
            work_dir_target=f'{mesh_path}/culled_mesh.nc')

        self.add_input_file(
            filename='graph.info',
            work_dir_target=f'{mesh_path}/culled_graph.info')

        if self.mesh.with_ice_shelf_cavities:
            self.add_input_file(
                filename='land_ice_mask.nc',
                work_dir_target=f'{mesh_path}/land_ice_mask.nc')

        self.add_model_as_input()

        for file in ['initial_state.nc', 'init_mode_forcing_data.nc',
                     'graph.info']:
            self.add_output_file(filename=file)

        if with_inactive_top_cells:
            self.add_output_file(filename='initial_state_crop.nc')

    def setup(self):
        """
        Get resources at setup from config options
        """
        self._get_resources()
        rx1_max = self.config.getfloat('global_ocean', 'rx1_max')
        self.add_namelist_options({'config_rx1_max': f'{rx1_max}'},
                                  mode='init')

    def constrain_resources(self, available_resources):
        """
        Update resources at runtime from config options
        """
        self._get_resources()
        super().constrain_resources(available_resources)

    def runtime_setup(self):
        """
        Update the Haney number at runtime based on the config option.
        """
        rx1_max = self.config.getfloat('global_ocean', 'rx1_max')
        self.update_namelist_at_runtime({'config_rx1_max': f'{rx1_max}'})

    def run(self):
        """
        Run this step of the testcase
        """
        config = self.config
        if self.with_inactive_top_cells:
            # Since we start at minLevelCell = 2, we need to increase the
            # number of vertical levels in the cfg file to end up with the
            # intended number in the initial state
            vert_levels = config.getint('vertical_grid', 'vert_levels')
            config.set('vertical_grid', 'vert_levels', f'{vert_levels + 1}',
                       comment='the number of vertical levels + 1')
            config.set('vertical_grid', 'inactive_top_cells', '1')
        logger = self.logger
        interfaces = generate_1d_grid(config=config)

        write_1d_grid(interfaces=interfaces, out_filename='vertical_grid.nc')
        plot_vertical_grid(grid_filename='vertical_grid.nc', config=config,
                           out_filename='vertical_grid.png')

        update_pio = config.getboolean('global_ocean', 'init_update_pio')
        run_model(self, update_pio=update_pio)

        if self.with_inactive_top_cells:

            logger.info("   * Updating minLevelCell for inactive top cells")

            in_filename = 'initial_state.nc'
            out_filename = in_filename

            with xarray.open_dataset(in_filename) as ds:
                ds.load()

                # keep the data set with Time for output
                ds_out = ds

                ds = ds.isel(Time=0)

                if config.has_option('vertical_grid', 'inactive_top_cells'):
                    offset = config.getint('vertical_grid',
                                           'inactive_top_cells')
                else:
                    offset = 0

                if 'minLevelCell' in ds:
                    minLevelCell = ds.minLevelCell + offset
                    ds_out['minLevelCell'] = minLevelCell
                else:
                    logger.info("   - Variable minLevelCell, needed for "
                                "inactive top cells, is missing from the "
                                "initial condition")

            write_netcdf(ds_out, out_filename)

            remove_inactive_top_cells_output(
                in_filename=in_filename, out_filename='initial_state_crop.nc')

            logger.info("   - Complete")

        add_mesh_and_init_metadata(self.outputs, config,
                                   init_filename='initial_state.nc')

        plot_initial_state(input_file_name='initial_state.nc',
                           output_file_name='initial_state.png')

    def _get_resources(self):
        # get the these properties from the config options
        config = self.config
        self.ntasks = config.getint('global_ocean', 'init_ntasks')
        self.min_tasks = config.getint('global_ocean', 'init_min_tasks')
        self.openmp_threads = config.getint('global_ocean', 'init_threads')
        self.cpus_per_task = config.getint('global_ocean',
                                           'init_cpus_per_task')
        self.min_cpus_per_task = self.cpus_per_task
