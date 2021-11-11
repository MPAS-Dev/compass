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
    def __init__(self, test_case, resolution, name='forward', subdir=None,
                 with_particles=False):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : str
            The resolution of the test case

        name : str
            the name of the test case

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        with_particles : bool, optional
            Whether to run with Lagrangian particles
        """
        self.resolution = resolution
        self.with_particles = with_particles
        res_params = {'32km': {'cores': 4,
                               'min_cores': 1,
                               'dt': "'00:10:00'",
                               'btr_dt': "'0000_00:00:25'",
                               'mom_del4': "2.0e11"},
                      '16km': {'cores': 4,
                               'min_cores': 1,
                               'dt': "'00:10:00'",
                               'btr_dt': "'0000_00:00:25'",
                               'mom_del4': "2.0e10 "},
                      '8km': {'cores': 4,
                              'min_cores': 1,
                              'dt': "'00:10:00'",
                              'btr_dt': "'0000_00:00:15'",
                              'mom_del4': "2.0e9"},
                      '4km': {'cores': 4,
                              'min_cores': 1,
                              'dt': "'00:06:00'",
                              'btr_dt': "'0000_00:00:10'",
                              'mom_del4': "4.0e8"}}

        if resolution not in res_params:
            raise ValueError(
                f'Unsupported resolution {resolution}. Supported values are: '
                f'{list(res_params)}')

        res_params = res_params[resolution]

        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         cores=res_params['cores'],
                         min_cores=res_params['min_cores'])
        # make sure output is double precision
        self.add_streams_file('compass.ocean.streams', 'streams.output')

        self.add_namelist_file('compass.ocean.tests.soma', 'namelist.forward')
        self.add_streams_file('compass.ocean.tests.soma', 'streams.forward')
        self.add_namelist_file('compass.ocean.tests.soma', 'namelist.analysis')
        self.add_streams_file('compass.ocean.tests.soma', 'streams.analysis')

        options = dict()
        for option in ['dt', 'btr_dt', 'mom_del4']:
            options[f'config_{option}'] = res_params[option]
        if with_particles:
            options['config_AM_lagrPartTrack_enable'] = '.true.'
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
        cores = self.cores
        partition(cores, self.config, self.logger)
        if self.with_particles:
            section = self.config['soma']
            min_den = section.getfloat('min_particle_density')
            max_den = section.getfloat('max_particle_density')
            nsurf = section.getint('surface_count')
            build_particle_simple(
                f_grid='mesh.nc', f_name='particles.nc',
                f_decomp='graph.info.part.{}'.format(cores),
                buoySurf=np.linspace(min_den, max_den, nsurf))
        run_model(self, partition_graph=False)
