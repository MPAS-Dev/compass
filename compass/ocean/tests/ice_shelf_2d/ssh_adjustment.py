import time

from compass.ocean.iceshelf import adjust_ssh
from compass.step import Step


class SshAdjustment(Step):
    """
    A step for iteratively adjusting the pressure from the weight of the ice
    shelf to match the sea-surface height as part of ice-shelf 2D test cases
    """
    def __init__(self, test_case, resolution, coord_type, ntasks=1,
                 min_tasks=None, openmp_threads=1, tidal_forcing=False):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : float
            The resolution of the test case in m

        coord_type: str
            The coordinate type (e.g., 'z-star', 'single_layer', etc.)

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
        self.resolution = resolution
        if min_tasks is None:
            min_tasks = ntasks
        super().__init__(test_case=test_case, name='ssh_adjustment',
                         ntasks=ntasks, min_tasks=min_tasks,
                         openmp_threads=openmp_threads)

        # generate the namelist, replacing a few default options
        # start with the same namelist settings as the forward run
        self.add_namelist_file('compass.ocean.tests.ice_shelf_2d',
                               'namelist.forward')
        if coord_type == 'single_layer':
            self.add_namelist_file(
                'compass.ocean.tests.ice_shelf_2d',
                'namelist.single_layer.forward_and_ssh_adjust')

        # we don't want the global stats AM for this run
        self.add_namelist_options({'config_AM_globalStats_enable': '.false.'})

        # we want a shorter run and no freshwater fluxes under the ice shelf
        # from these namelist options
        self.add_namelist_file('compass.ocean.namelists',
                               'namelist.ssh_adjust')

        self.add_streams_file('compass.ocean.streams', 'streams.ssh_adjust')

        self.add_input_file(filename='adjusting_init0.nc',
                            target='../initial_state/initial_state.nc')

        self.add_input_file(filename='graph.info',
                            target='../initial_state/culled_graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='adjusted_init.nc')

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        dt_per_km = config.getfloat('ice_shelf_2d', 'dt_per_km')
        dt = dt_per_km * self.resolution / 1.e3
        dt_str = time.strftime('%H:%M:%S', time.gmtime(dt))
        self.update_namelist_at_runtime({'config_dt': dt_str})
        iteration_count = config.getint('ssh_adjustment', 'iterations')
        adjust_ssh(variable='landIcePressure', iteration_count=iteration_count,
                   step=self, delta_ssh_threshold=1.e-10)
