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
                 cores=1, min_cores=None, threads=1, with_frazil=True):
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

        with_frazil : bool, optional
            whether the simulation includes frazil ice formation
        """
        self.resolution = resolution
        if min_cores is None:
            min_cores = cores
        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         cores=cores, min_cores=min_cores, threads=threads)

        self.add_namelist_file('compass.ocean.tests.ice_shelf_2d',
                               'namelist.forward')
        if with_frazil:
            options = {'config_use_frazil_ice_formation': '.true.',
                       'config_frazil_maximum_depth': '2000.0'}
            self.add_namelist_options(options)
            self.add_streams_file('compass.ocean.streams', 'streams.frazil')

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

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """
        run_model(self)
