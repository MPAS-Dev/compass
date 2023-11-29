from compass.model import run_model
from compass.step import Step


class Forward(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of dam break
    test cases.
    """
    def __init__(self, test_case, resolution, use_lts,
                 name='forward', subdir=None,
                 ntasks=1, min_tasks=None, openmp_threads=1):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : str
            The resolution of the test case

        use_lts : bool
            Whether local time-stepping is used

        name : str
            the name of the test case

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        ntasks: int, optional
            the number of tasks the step would ideally use.  If fewer tasks
            are available on the system, the step will run on all available
            tasks as long as this is not below ``min_tasks``

        min_tasks: int, optional
            the number of tasks the step requires.  If the system has fewer
            than this number of tasks, the step will fail

        openmp_threads : int, optional
            the number of threads the step will use

        """
        if min_tasks is None:
            min_tasks = ntasks

        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         ntasks=ntasks, min_tasks=min_tasks,
                         openmp_threads=openmp_threads)

        self.resolution = resolution

        self.add_namelist_file('compass.ocean.tests.dam_break',
                               'namelist.forward')

        if use_lts:
            self.add_namelist_options(
                {'config_time_integrator': "'LTS'"})
            self.add_namelist_options(
                {'config_dt_scaling_LTS': "4"})
            self.add_namelist_options(
                {'config_number_of_time_levels': "4"})

            self.add_streams_file('compass.ocean.tests.dam_break.lts',
                                  'streams.forward')
            input_path = '../lts_regions'
            self.add_input_file(filename='mesh.nc',
                                target=f'{input_path}/lts_mesh.nc')
            self.add_input_file(filename='graph.info',
                                target=f'{input_path}/lts_graph.info')
            self.add_input_file(filename='init.nc',
                                target=f'{input_path}/lts_ocean.nc')

        else:
            self.add_streams_file('compass.ocean.tests.dam_break',
                                  'streams.forward')
            input_path = '../initial_state'
            self.add_input_file(filename='mesh.nc',
                                target=f'{input_path}/culled_mesh.nc')

            self.add_input_file(filename='init.nc',
                                target=f'{input_path}/ocean.nc')

            self.add_input_file(filename='graph.info',
                                target=f'{input_path}/culled_graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='output.nc')

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """

        resolution = self.resolution
        if resolution == 0.04:
            self.update_namelist_at_runtime({'config_dt':
                                             "'0000_00:00:00.001'"})
        elif resolution == 0.12:
            self.update_namelist_at_runtime({'config_dt':
                                             "'0000_00:00:00.003'"})
        run_model(self)
