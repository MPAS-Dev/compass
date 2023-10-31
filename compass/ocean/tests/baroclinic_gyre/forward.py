from compass.model import run_model
from compass.step import Step


class Forward(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of
    the baroclinic gyre test cases.

    Attributes
    ----------
    resolution : str
        The resolution of the test case

    """
    def __init__(self, test_case, resolution, name='forward', subdir=None,
                 long=False):
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

        long : bool, optional
            Whether to run a long (3-year) simulation to quasi-equilibrium
        """
        self.resolution = resolution
        res_params = {'80km': {'ntasks': 20,  # MODIFY
                               'min_tasks': 2,
                               'dt': "'00:12:00'",
                               'btr_dt': "'00:00:36'",
                               'mom_del4': "5.0e10",
                               'run_duration': "'0000_00:36:00'"}}

        if resolution not in res_params:
            raise ValueError(
                f'Unsupported resolution {resolution}. Supported values are: '
                f'{list(res_params)}')

        res_params = res_params[resolution]

        ntasks = res_params['ntasks']
        min_tasks = res_params['min_tasks']

        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         ntasks=ntasks, min_tasks=min_tasks, openmp_threads=1)

        # make sure output is double precision
        self.add_streams_file('compass.ocean.streams', 'streams.output')

        self.add_namelist_file('compass.ocean.tests.baroclinic_gyre',
                               'namelist.forward')
        if long:
            output_interval = "0010_00:00:00"
            restart_interval = "0010_00:00:00"
        else:
            output_interval = res_params['run_duration'].replace("'", "")
            restart_interval = "0030_00:00:00"
        replacements = dict(
            output_interval=output_interval, restart_interval=restart_interval)
        self.add_streams_file(package='compass.ocean.tests.baroclinic_gyre',
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

        self.add_input_file(filename='init.nc',
                            target='../initial_state/initial_state.nc')
        self.add_input_file(filename='forcing.nc',
                            target='../initial_state/forcing.nc')
        self.add_input_file(filename='graph.info',
                            target='../initial_state/culled_graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='output/output.0001-01-01_00.00.00.nc')

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """
        run_model(self, partition_graph=True)
