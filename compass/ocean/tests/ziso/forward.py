from compass.step import Step
from compass.model import partition, run_model
from compass.ocean import particles


class Forward(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of ZISO test cases.

    Attributes
    ----------
    resolution : str
        The resolution of the test case

    with_analysis : bool, optional
        whether analysis members are enabled as part of the run

    with_frazil : bool, optional
        whether the run includes frazil formation
    """
    def __init__(self, test_case, resolution, name='forward', subdir=None,
                 with_analysis=False, with_frazil=False, long=False,
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

        with_analysis : bool, optional
            whether analysis members are enabled as part of the run

        with_frazil : bool, optional
            whether the run includes frazil formation

        long : bool, optional
            Whether to run a long (3-year) simulation to quasi-equilibrium

        with_particles : bool, optional
            Whether particles are include in the simulation
        """
        self.with_particles = with_particles
        self.resolution = resolution
        self.with_analysis = with_analysis
        self.with_frazil = with_frazil
        res_params = {'20km': {'ntasks': 20,
                               'min_tasks': 2,
                               'ntasks_with_particles': 32,
                               'min_tasks_with_particles': 12,
                               'dt': "'00:12:00'",
                               'btr_dt': "'00:00:36'",
                               'mom_del4': "5.0e10",
                               'run_duration': "'0000_00:36:00'"},
                      '10km': {'ntasks': 80,
                               'min_tasks': 8,
                               'ntasks_with_particles': 130,
                               'min_tasks_with_particles': 50,
                               'dt': "'00:06:00'",
                               'btr_dt': "'00:00:18'",
                               'mom_del4': "6.25e9",
                               'run_duration': "'0000_00:18:00'"},
                      '5km': {'ntasks': 300,
                              'min_tasks': 30,
                              'ntasks_with_particles': 500,
                              'min_tasks_with_particles': 200,
                              'dt': "'00:03:00'",
                              'btr_dt': "'00:00:09'",
                              'mom_del4': "7.8e8",
                              'run_duration': "'0000_00:09:00'"},
                      '2.5km': {'ntasks': 1200,
                                'min_tasks': 120,
                                'ntasks_with_particles': 2100,
                                'min_tasks_with_particles': 900,
                                'dt': "'00:01:30'",
                                'btr_dt': "'00:00:04'",
                                'mom_del4': "9.8e7",
                                'run_duration': "'0000_00:04:30'"}}

        if resolution not in res_params:
            raise ValueError(
                f'Unsupported resolution {resolution}. Supported values are: '
                f'{list(res_params)}')

        res_params = res_params[resolution]

        if with_particles:
            ntasks = res_params['ntasks_with_particles']
            min_tasks = res_params['min_tasks_with_particles']
        else:
            ntasks = res_params['ntasks']
            min_tasks = res_params['min_tasks']

        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         ntasks=ntasks, min_tasks=min_tasks, openmp_threads=1)

        # make sure output is double precision
        self.add_streams_file('compass.ocean.streams', 'streams.output')

        self.add_namelist_file('compass.ocean.tests.ziso', 'namelist.forward')
        if long:
            output_interval = "0010_00:00:00"
            restart_interval = "0010_00:00:00"
        else:
            output_interval = res_params['run_duration'].replace("'", "")
            restart_interval = "0030_00:00:00"
        replacements = dict(
            output_interval=output_interval, restart_interval=restart_interval)
        self.add_streams_file(package='compass.ocean.tests.ziso',
                              streams='streams.forward',
                              template_replacements=replacements)
        options = dict()
        for option in ['dt', 'btr_dt', 'mom_del4', 'run_duration']:
            options[f'config_{option}'] = res_params[option]
        if long:
            # run for 3 years instead of 3 time steps
            options['config_start_time'] = "'0001-01-01_00:00:00'"
            options['config_stop_time'] = "'0004-01-01_00:00:00'"
            options['config_run_duration'] = "'none'"

        self.add_namelist_options(options=options)
        if with_analysis:
            self.add_namelist_file('compass.ocean.tests.ziso',
                                   'namelist.analysis')
            self.add_streams_file('compass.ocean.tests.ziso',
                                  'streams.analysis')

        if with_particles:
            self.add_namelist_file('compass.ocean.tests.ziso',
                                   'namelist.particles')
            self.add_streams_file('compass.ocean.tests.ziso',
                                  'streams.particles')

        if with_frazil:
            self.add_namelist_options(
                {'config_use_frazil_ice_formation': '.true.'})
            self.add_streams_file('compass.ocean.streams', 'streams.frazil')
            self.add_output_file('frazil.nc')

        self.add_input_file(filename='init.nc',
                            target='../initial_state/ocean.nc')
        self.add_input_file(filename='forcing.nc',
                            target='../initial_state/forcing.nc')
        self.add_input_file(filename='graph.info',
                            target='../initial_state/culled_graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='output/output.0001-01-01_00.00.00.nc')

        if with_analysis and with_particles:
            self.add_output_file(
                filename='analysis_members/lagrPartTrack.0001-01-01_00.00.00.nc')

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """
        if self.with_particles:
            ntasks = self.ntasks
            partition(ntasks, self.config, self.logger)
            particles.write(init_filename='init.nc',
                            particle_filename='particles.nc',
                            graph_filename=f'graph.info.part.{ntasks}',
                            types='buoyancy')
            run_model(self, partition_graph=False)
        else:
            run_model(self, partition_graph=True)
