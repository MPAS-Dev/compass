from compass.model import run_model
from compass.step import Step


class Forward(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of ice-shelf 2D test
    cases.

    Attributes
    ----------
    resolution : str
        The resolution of the test case
    """
    def __init__(self, test_case, resolution, name='forward', subdir=None,
                 ntasks=1, min_tasks=None, openmp_threads=1, with_frazil=True):
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

        with_frazil : bool, optional
            whether the simulation includes frazil ice formation
        """
        self.resolution = resolution
        if min_tasks is None:
            min_tasks = ntasks
        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         ntasks=ntasks, min_tasks=min_tasks,
                         openmp_threads=openmp_threads)

        # make sure output is double precision
        self.add_streams_file('compass.ocean.streams', 'streams.output')

        self.add_namelist_file('compass.ocean.tests.ice_shelf_2d',
                               'namelist.forward')
        if with_frazil:
            options = {'config_use_frazil_ice_formation': '.true.',
                       'config_frazil_maximum_depth': '2000.0'}
            self.add_namelist_options(options)
            self.add_streams_file('compass.ocean.streams', 'streams.frazil')
            self.add_output_file('frazil.nc')

        self.add_streams_file('compass.ocean.streams',
                              'streams.land_ice_fluxes')

        self.add_streams_file('compass.ocean.tests.ice_shelf_2d',
                              'streams.forward')

        self.add_input_file(filename='init.nc',
                            target='../ssh_adjustment/adjusted_init.nc')
        self.add_input_file(filename='graph.info',
                            target='../initial_state/culled_graph.info')

        self.add_model_as_input()

        self.add_output_file('output.nc')
        self.add_output_file('land_ice_fluxes.nc')

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """
        run_model(self)
