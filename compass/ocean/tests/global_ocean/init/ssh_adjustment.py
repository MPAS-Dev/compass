from compass.step import Step
from compass.ocean.iceshelf import adjust_ssh


class SshAdjustment(Step):
    """
    A step for iteratively adjusting the pressure from the weight of the ice
    shelf to match the sea-surface height as part of ice-shelf 2D test cases
    """
    def __init__(self, test_case, ntasks=None, min_tasks=None,
                 openmp_threads=None):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.init.Init
            The test case this step belongs to

        ntasks : int, optional
            the number of tasks the step would ideally use.  If fewer tasks
            are available on the system, the step will run on all available
            tasks as long as this is not below ``min_tasks``

        min_tasks : int, optional
            the number of tasks the step requires.  If the system has fewer
            than this number of tasks, the step will fail

        openmp_threads : int, optional
            the number of OpenMP threads the step will use

        """
        if min_tasks is None:
            min_tasks = ntasks
        super().__init__(test_case=test_case, name='ssh_adjustment',
                         ntasks=ntasks, min_tasks=min_tasks,
                         openmp_threads=openmp_threads)

        # make sure output is double precision
        self.add_streams_file('compass.ocean.streams', 'streams.output')

        self.add_namelist_file(
            'compass.ocean.tests.global_ocean', 'namelist.forward')
        self.add_namelist_options({'config_AM_globalStats_enable': '.false.'})
        self.add_namelist_file('compass.ocean.namelists',
                               'namelist.ssh_adjust')

        self.add_streams_file('compass.ocean.streams', 'streams.ssh_adjust')
        self.add_streams_file('compass.ocean.tests.global_ocean.init',
                              'streams.ssh_adjust')

        mesh_path = test_case.mesh.mesh_step.path
        init_path = test_case.steps['initial_state'].path

        self.add_input_file(
            filename='adjusting_init0.nc',
            work_dir_target='{}/initial_state.nc'.format(init_path))
        self.add_input_file(
            filename='forcing_data.nc',
            work_dir_target='{}/init_mode_forcing_data.nc'.format(init_path))
        self.add_input_file(
            filename='graph.info',
            work_dir_target='{}/culled_graph.info'.format(mesh_path))

        self.add_model_as_input()

        self.add_output_file(filename='adjusted_init.nc')

    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies
        """
        if self.ntasks is None:
            self.ntasks = self.config.getint(
                'global_ocean', 'forward_ntasks')
        if self.min_tasks is None:
            self.min_tasks = self.config.getint(
                'global_ocean', 'forward_min_tasks')
        if self.openmp_threads is None:
            self.openmp_threads = self.config.getint(
                'global_ocean', 'forward_threads')

    def run(self):
        """
        Run this step of the testcase
        """
        config = self.config
        iteration_count = config.getint('ssh_adjustment', 'iterations')
        adjust_ssh(variable='landIcePressure', iteration_count=iteration_count,
                   step=self)
