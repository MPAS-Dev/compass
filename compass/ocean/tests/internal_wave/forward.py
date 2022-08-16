from compass.model import run_model
from compass.step import Step


class Forward(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of internal wave
    test cases.
    """
    def __init__(self, test_case, name='forward', subdir=None, ntasks=1,
                 min_tasks=None, openmp_threads=1, nu=None):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

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

        nu : float, optional
            the viscosity (if different from the default for the test group)
        """
        if min_tasks is None:
            min_tasks = ntasks
        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         ntasks=ntasks, min_tasks=min_tasks,
                         openmp_threads=openmp_threads)
        self.add_namelist_file('compass.ocean.tests.internal_wave',
                               'namelist.forward')
        if nu is not None:
            # update the viscosity to the requested value
            options = {'config_mom_del2': '{}'.format(nu)}
            self.add_namelist_options(options)

        self.add_streams_file('compass.ocean.tests.internal_wave',
                              'streams.forward')

        self.add_input_file(filename='init.nc',
                            target='../initial_state/ocean.nc')
        self.add_input_file(filename='mesh.nc',
                            target='../initial_state/culled_mesh.nc')
        self.add_input_file(filename='graph.info',
                            target='../initial_state/culled_graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='output.nc')

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """
        run_model(self)
