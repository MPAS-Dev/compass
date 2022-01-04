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
                 cores=1, min_cores=None, threads=1, with_analysis=False,
                 with_frazil=False, long=False):
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

        cores : int, optional
            the number of cores the step would ideally use.  If fewer cores
            are available on the system, the step will run on all available
            cores as long as this is not below ``min_cores``

        min_cores : int, optional
            the number of cores the step requires.  If the system has fewer
            than this number of cores, the step will fail

        threads : int, optional
            the number of threads the step will use

        with_analysis : bool, optional
            whether analysis members are enabled as part of the run

        with_frazil : bool, optional
            whether the run includes frazil formation

        long : bool
            Whether to run a long (3-year) simulation to quasi-equilibrium
        """
        self.resolution = resolution
        self.with_analysis = with_analysis
        self.with_frazil = with_frazil
        res_params = {'20km': {'cores': 20,
                               'min_cores': 2,
                               'dt': "'00:12:00'",
                               'btr_dt': "'00:00:36'",
                               'mom_del4': "5.0e10",
                               'run_duration': "'0000_00:36:00'"},
                      '10km': {'cores': 80,
                               'min_cores': 8,
                               'dt': "'00:06:00'",
                               'btr_dt': "'00:00:18'",
                               'mom_del4': "6.25e9",
                               'run_duration': "'0000_00:18:00'"},
                      '5km': {'cores': 300,
                              'min_cores': 30,
                              'dt': "'00:03:00'",
                              'btr_dt': "'00:00:09'",
                              'mom_del4': "7.8e8",
                              'run_duration': "'0000_00:09:00'"},
                      '2.5km': {'cores': 1200,
                                'min_cores': 120,
                                'dt': "'00:01:30'",
                                'btr_dt': "'00:00:04'",
                                'mom_del4': "9.8e7",
                                'run_duration': "'0000_00:04:30"}}

        if min_cores is None:
            min_cores = cores
        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         cores=cores, min_cores=min_cores, threads=threads)

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

        if with_analysis:
            self.add_namelist_file('compass.ocean.tests.ziso',
                                   'namelist.analysis')
            self.add_streams_file('compass.ocean.tests.ziso',
                                  'streams.analysis')

        if with_frazil:
            self.add_namelist_options(
                {'config_use_frazil_ice_formation': '.true.'})
            self.add_streams_file('compass.ocean.streams', 'streams.frazil')
            self.add_output_file('frazil.nc')

        self.add_namelist_file('compass.ocean.tests.ziso',
                               'namelist.{}.forward'.format(resolution))
        self.add_streams_file('compass.ocean.tests.ziso',
                              'streams.{}.forward'.format(resolution))

        self.add_input_file(filename='init.nc',
                            target='../initial_state/ocean.nc')
        self.add_input_file(filename='forcing.nc',
                            target='../initial_state/forcing.nc')
        self.add_input_file(filename='graph.info',
                            target='../initial_state/culled_graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='output/output.0001-01-01_00.00.00.nc')

        if with_analysis:
            self.add_output_file(
                filename='analysis_members/lagrPartTrack.0001-01-01_00.00.00.nc')

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """
        cores = self.cores
        partition(cores, self.config, self.logger)
        particles.write(init_filename='init.nc', particle_filename='particles.nc',
                        graph_filename='graph.info.part.{}'.format(cores),
                        types='buoyancy')
        run_model(self, partition_graph=False)
