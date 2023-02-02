from compass.model import run_model
from compass.step import Step


class Forward(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of the
    solitary wave test case.

    Attributes
    ----------
    nonhydro_mode : bool
        The resolution of the test case
    """
    def __init__(self, test_case, nonhydro_mode, name,
                 ntasks=16, min_tasks=1, openmp_threads=1):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        name : str
            the name of the test case

        ntasks : int, optional
            the number of cores the step would ideally use.  If fewer cores
            are available on the system, the step will run on all available
            cores as long as this is not below ``min_cores``

        min_tasks : int, optional
            the number of cores the step requires.  If the system has fewer
            than this number of cores, the step will fail

        openmp_threads : int, optional
            the number of threads the step will use

        """
        self.nonhydro_mode = nonhydro_mode
        super().__init__(test_case=test_case, name=name,
                         ntasks=ntasks, min_tasks=min_tasks,
                         openmp_threads=openmp_threads)
        self.add_namelist_file('compass.ocean.tests.nonhydro.solitary_wave',
                               'namelist.forward')
        if nonhydro_mode:
            self.add_namelist_file(
                'compass.ocean.tests.nonhydro.solitary_wave',
                'namelist.nonhydro')
        else:
            self.add_namelist_file(
                'compass.ocean.tests.nonhydro.solitary_wave',
                'namelist.hydro')

        # make sure output is double precision
        self.add_streams_file('compass.ocean.streams', 'streams.output')

        self.add_streams_file('compass.ocean.tests.nonhydro.solitary_wave',
                              'streams.forward')

        self.add_input_file(filename='init.nc',
                            target='../initial_state/initial_state.nc')
        self.add_input_file(filename='graph.info',
                            target='../initial_state/culled_graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='output.nc')

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """
        run_model(self, update_pio=False)
