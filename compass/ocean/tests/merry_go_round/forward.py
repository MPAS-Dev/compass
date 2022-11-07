from compass.model import run_model
from compass.step import Step


class Forward(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of merry-go-round
    test cases.

    Attributes
    ----------
    resolution : str
        The resolution of the test case
    """
    def __init__(self, test_case, resolution, name='forward', subdir=None,
                 ntasks=1, min_tasks=None, openmp_threads=1):
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
        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         ntasks=ntasks, min_tasks=min_tasks,
                         openmp_threads=openmp_threads)
        self.add_namelist_file('compass.ocean.tests.merry_go_round',
                               'namelist.forward')

        # make sure output is double precision
        self.add_streams_file('compass.ocean.tests.merry_go_round',
                              'streams.forward')

        self.add_input_file(filename='init.nc',
                            target=f'../initial_state_{resolution}/init.nc')
        self.add_input_file(filename='graph.info',
                            target=f'../initial_state_{resolution}/culled_graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='output.nc')

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """
        run_params = {'5m':    {'config_dt': '00:12:00'},
                      '2.5m':  {'config_dt': '00:06:00'},
                      '1.25m': {'config_dt': '00:03:00'}}
        self.update_namelist_at_runtime(run_params[self.resolution])
        run_model(self)
