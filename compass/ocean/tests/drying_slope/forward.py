import time

from compass.model import run_model
from compass.step import Step


class Forward(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of drying slope
    test cases.
    """
    def __init__(self, test_case, resolution, name='forward', subdir=None,
                 input_path='../initial_state',
                 ntasks=1, min_tasks=None, openmp_threads=1,
                 damping_coeff=None, coord_type='sigma'):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : float
            The resolution of the test case

        name : str
            the name of the test case

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        ntasks : int, optional
            the number of tasks the step would ideally use.  If fewer tasks
            are available on the system, the step will run on all available
            tasks as long as this is not below ``min_tasks``

        min_tasks : int, optional
            the number of tasks the step requires.  If the system has fewer
            than this number of tasks, the step will fail

        openmp_threads : int, optional
            the number of OpenMP threads the step will use

        damping_coeff: float, optional
            the value of the rayleigh damping coefficient

        coord_type: string, optional
            the coordinate type

        """
        if min_tasks is None:
            min_tasks = ntasks

        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         ntasks=ntasks, min_tasks=min_tasks,
                         openmp_threads=openmp_threads)

        self.resolution = resolution
        self.add_namelist_file('compass.ocean.tests.drying_slope',
                               'namelist.forward')
        if coord_type == 'single_layer' or coord_type == 'sigma':
            self.add_namelist_file('compass.ocean.tests.drying_slope',
                                   f'namelist.{coord_type}.forward')
        if damping_coeff is not None:
            # update the Rayleigh damping coeff to the requested value
            options = {'config_Rayleigh_damping_coeff': f'{damping_coeff}'}
            self.add_namelist_options(options)

        self.add_streams_file('compass.ocean.tests.drying_slope',
                              'streams.forward')

        self.add_input_file(filename='mesh.nc',
                            target=f'{input_path}/culled_mesh.nc')

        self.add_input_file(filename='init.nc',
                            target=f'{input_path}/ocean.nc')

        self.add_input_file(filename='forcing.nc',
                            target=f'{input_path}/init_mode_forcing_data.nc')

        self.add_input_file(filename='graph.info',
                            target=f'{input_path}/culled_graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='output.nc')

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """
        dt = self.get_dt()
        self.update_namelist_at_runtime(options={'config_dt': f"'{dt}'"},
                                        out_name='namelist.ocean')

        run_model(self)

    def get_dt(self):
        """
        Get the time step

        Returns
        -------
        dt : str
            the time step in HH:MM:SS
        """
        config = self.config
        # dt is proportional to resolution
        dt_per_km = config.getfloat('drying_slope', 'dt_per_km')

        dt = dt_per_km * self.resolution
        # https://stackoverflow.com/a/1384565/7728169
        dt = time.strftime('%H:%M:%S', time.gmtime(dt))

        return dt
