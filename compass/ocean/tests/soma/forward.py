import numpy as np

from compass.step import Step
from compass.model import partition, run_model
from compass.ocean.particles import build_particle_simple


class Forward(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of SOMA test cases.

    Attributes
    ----------
    resolution : str
        The resolution of the test case

    with_particles : bool
        Whether to run with Lagrangian particles
    """
    def __init__(self, test_case, resolution, with_particles,
                 with_surface_restoring, long, three_layer):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : str
            The resolution of the test case

        with_particles : bool, optional
            Whether to run with Lagrangian particles

        with_surface_restoring : bool
            Whether surface restoring is included in the simulation

        long : bool
            Whether to run a long (3-year) simulation to quasi-equilibrium

        three_layer : bool
            Whether to use only 3 vertical layers and no continental shelf
        """
        self.resolution = resolution
        self.with_particles = with_particles
        res_params = {'32km': {'cores': 25,
                               'min_tasks': 3,
                               'dt': "'00:24:00'",
                               'btr_dt': "'0000_00:00:48'",
                               'mom_del4': "2.0e11",
                               'run_duration': "'0000_02:00:00'"},
                      '16km': {'cores': 100,
                               'min_tasks': 10,
                               'dt': "'00:12:00'",
                               'btr_dt': "'0000_00:00:24'",
                               'mom_del4': "2.0e10 ",
                               'run_duration': "'0000_01:00:00'"},
                      '8km': {'cores': 400,
                              'min_tasks': 40,
                              'dt': "'00:06:00'",
                              'btr_dt': "'0000_00:00:12'",
                              'mom_del4': "2.0e9",
                              'run_duration': "'0000_00:30:00'"},
                      '4km': {'cores': 1600,
                              'min_tasks': 160,
                              'dt': "'00:03:00'",
                              'btr_dt': "'0000_00:00:06'",
                              'mom_del4': "4.0e8",
                              'run_duration': "'0000_00:15:00'"}}

        if resolution not in res_params:
            raise ValueError(
                f'Unsupported resolution {resolution}. Supported values are: '
                f'{list(res_params)}')

        res_params = res_params[resolution]

        super().__init__(test_case=test_case, name='forward', subdir=None,
                         ntasks=res_params['cores'],
                         min_tasks=res_params['min_tasks'])
        # make sure output is double precision
        self.add_streams_file('compass.ocean.streams', 'streams.output')

        self.add_namelist_file('compass.ocean.tests.soma', 'namelist.forward')

        if long:
            output_interval = "0010_00:00:00"
            restart_interval = "0010_00:00:00"
        else:
            output_interval = res_params['run_duration'].replace("'", "")
            restart_interval = "0030_00:00:00"
        replacements = dict(
            output_interval=output_interval, restart_interval=restart_interval)
        self.add_streams_file(package='compass.ocean.tests.soma',
                              streams='streams.forward',
                              template_replacements=replacements)
        self.add_namelist_file('compass.ocean.tests.soma', 'namelist.analysis')
        self.add_streams_file('compass.ocean.tests.soma', 'streams.analysis')

        options = dict()
        for option in ['dt', 'btr_dt', 'mom_del4', 'run_duration']:
            options[f'config_{option}'] = res_params[option]
        if with_particles:
            options['config_AM_lagrPartTrack_enable'] = '.true.'
        if long:
            # run for 3 years instead of 3 time steps
            options['config_start_time'] = "'0001-01-01_00:00:00'"
            options['config_stop_time'] = "'0004-01-01_00:00:00'"
            options['config_run_duration'] = "'none'"

        if three_layer:
            # set config options for 3-layer run instead of 60 layers
            options['config_vert_coord_movement'] = "'impermeable_interfaces'"
            options['config_use_cvmix'] = 'false.'
            options['config_AM_mixedLayerDepths_enable'] = 'false.'

        if with_surface_restoring:
            options['config_use_activeTracers_surface_restoring'] = '.true.'

        self.add_namelist_options(options=options)

        self.add_input_file(filename='mesh.nc',
                            target='../initial_state/culled_mesh.nc')
        self.add_input_file(filename='init.nc',
                            target='../initial_state/initial_state.nc')
        self.add_input_file(filename='forcing.nc',
                            target='../initial_state/forcing.nc')
        self.add_input_file(filename='graph.info',
                            target='../initial_state/graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='output/output.0001-01-01_00.00.00.nc')

        if with_particles:
            self.add_output_file(
                filename='analysis_members/lagrPartTrack.0001-01-01_00.00.00.nc')

    def run(self):
        """
        Run this step of the test case
        """
        ntasks = self.ntasks
        partition(ntasks, self.config, self.logger)

        if self.with_particles:
            section = self.config['soma']
            min_den = section.getfloat('min_particle_density')
            max_den = section.getfloat('max_particle_density')
            nsurf = section.getint('surface_count')
            build_particle_simple(
                f_grid='mesh.nc', f_name='particles.nc',
                f_decomp=f'graph.info.part.{ntasks}',
                buoySurf=np.linspace(min_den, max_den, nsurf))

        run_model(self, partition_graph=False)
