import os
from importlib.resources import contents, read_text

import numpy as np
import xarray as xr
from jinja2 import Template
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call

from compass.io import symlink
from compass.model import run_model
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

    adjustment_fraction : float
        The fraction of the way through iterative ssh adjustment for this
        step
    """
    def __init__(self, test_case, mesh, initial_condition,
                 culled_topo_path=None, name='initial_state', subdir=None,
                 adjustment_fraction=None):
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

        culled_topo_path : str, optional
            The path to a step where ``topography_culled.nc`` is provided

        name : str, optional
            The name of the step

        subdir : str, optional
            The subdirectory for the step

        adjustment_fraction : float, optional
            The fraction of the way through iterative ssh adjustment for this
            step
        """
        if initial_condition not in ['WOA23', 'PHC', 'EN4_1900']:
            raise ValueError(f'Unknown initial_condition {initial_condition}')

        super().__init__(test_case=test_case, name=name, subdir=subdir)
        self.mesh = mesh
        self.initial_condition = initial_condition
        if mesh.with_ice_shelf_cavities and adjustment_fraction is None:
            raise ValueError('Must provide adjustment_fraction for '
                             'initializing meshes with ice-shelf cavities')
        self.adjustment_fraction = adjustment_fraction

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

        if culled_topo_path is None:
            cull_step = self.mesh.steps['cull_mesh']
            culled_topo_path = cull_step.path
        target = os.path.join(culled_topo_path, 'topography_culled.nc')
        self.add_input_file(filename='topography_culled.nc',
                            work_dir_target=target)

        self.add_input_file(
            filename='wind_stress.nc',
            target='windStress.ncep_1958-2000avg.interp3600x2431.151106.nc',
            database='initial_condition_database')

        if initial_condition == 'WOA23':
            self.add_input_file(
                filename='woa23.nc',
                target='woa23_decav_0.25_jan_extrap.20230416.nc',
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
        section = config['global_ocean']
        topo_filename = self._smooth_topography()

        interfaces = generate_1d_grid(config=config)

        write_1d_grid(interfaces=interfaces, out_filename='vertical_grid.nc')
        plot_vertical_grid(grid_filename='vertical_grid.nc', config=config,
                           out_filename='vertical_grid.png')

        min_levels = config.getint('global_ocean', 'min_levels')
        # minimum depth will be the depth of the middle of the minimum layer
        # so that MPAS-Ocean init mode will convert it back to the same
        # minimm number of levels
        min_depth = 0.5 * (interfaces[min_levels - 1] +
                           interfaces[min_levels])
        namelist = {'config_global_ocean_minimum_depth': f'{min_depth}'}

        if self.mesh.with_ice_shelf_cavities:
            frac = self.adjustment_fraction
            cavity_min_levels = section.getint('cavity_min_levels')
            min_thick_init = section.getfloat(
                'cavity_min_layer_thickness_initial')
            min_thick_final = section.getfloat(
                'cavity_min_layer_thickness_final')
            cavity_min_layer_thickness = (
                (1.0 - frac) * min_thick_init + frac * min_thick_final)
            namelist['config_rx1_min_levels'] = f'{cavity_min_levels}'
            namelist['config_rx1_min_layer_thickness'] = \
                f'{cavity_min_layer_thickness}'

            min_water_column_thickness = \
                cavity_min_layer_thickness * cavity_min_levels

            topo_filename = self._dig_cavity_bed_elevation(
                topo_filename, min_water_column_thickness)

        self.update_namelist_at_runtime(namelist)

        symlink(target=topo_filename, link_name='topography.nc')

        update_pio = config.getboolean('global_ocean', 'init_update_pio')
        run_model(self, update_pio=update_pio)

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

    def _smooth_topography(self):
        """ Smooth the topography using a Gaussian filter """
        config = self.config
        section = config['global_ocean']
        num_passes = section.getint('topo_smooth_num_passes')
        if num_passes == 0:
            # just return the culled topography file name
            return 'topography_culled.nc'

        distance_limit = section.getfloat('topo_smooth_distance_limit')
        std_deviation = section.getfloat('topo_smooth_std_deviation')

        template = Template(read_text(
            'compass.ocean.tests.global_ocean.init', 'smooth_topo.template'))

        text = template.render(num_passes=f'{num_passes}',
                               distance_limit=f'{distance_limit}',
                               std_deviation=f'{std_deviation}')

        # add trailing end line
        text = f'{text}\n'

        with open('smooth_depth_in', 'w') as file:
            file.write(text)

        check_call(args=['ocean_smooth_topo_before_init'],
                   logger=self.logger)

        out_filename = 'topography_smoothed.nc'
        with xr.open_dataset('topography_culled.nc') as ds_topo:
            with xr.open_dataset('topography_orig_and_smooth.nc') as ds_smooth:
                for field in ['bed_elevation', 'landIceDraftObserved',
                              'landIceThkObserved']:
                    attrs = ds_topo[field].attrs
                    ds_topo[field] = ds_smooth[f'{field}New']
                    ds_topo[field].attrs = attrs

            write_netcdf(ds_topo, out_filename)
        return out_filename

    def _dig_cavity_bed_elevation(self, in_filename,
                                  min_water_column_thickness):
        """ Dig bed elevation to preserve minimum water-column thickness """

        out_filename = 'topography_dig_bed.nc'
        with xr.open_dataset(in_filename) as ds_topo:
            bed = ds_topo.bed_elevation
            attrs = bed.attrs
            draft = ds_topo.landIceDraftObserved
            max_bed = draft - min_water_column_thickness
            mask = np.logical_or(draft == 0., bed < max_bed)
            bed = xr.where(mask, bed, max_bed)
            ds_topo['bed_elevation'] = bed
            ds_topo['bed_elevation'].attrs = attrs

            write_netcdf(ds_topo, out_filename)
        return out_filename
