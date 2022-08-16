from compass.config import CompassConfigParser
from compass.testcase import TestCase
from compass.ocean.tests.planar_convergence.forward import \
    Forward


class ConvTestCase(TestCase):
    """
    A test case for various convergence tests on in MPAS-Ocean with planar,
    doubly periodic meshes

    Attributes
    ----------
    resolutions : list of int
    """
    def __init__(self, test_group, name):
        """
        Create test case for creating a global MPAS-Ocean mesh

        Parameters
        ----------
        test_group : compass.ocean.tests.cosine_bell.GlobalOcean
            The global ocean test group that this test case belongs to

        name : str
            The name of the test case
        """
        super().__init__(test_group=test_group, name=name)
        self.resolutions = None

        # add the steps with default resolutions so they can be listed
        config = CompassConfigParser()
        module = 'compass.ocean.tests.planar_convergence'
        config.add_from_package(module, 'planar_convergence.cfg')
        self._setup_steps(config)

    def configure(self):
        """
        Set config options for the test case
        """
        config = self.config
        # set up the steps again in case a user has provided new resolutions
        self._setup_steps(config)

        self.update_cores()

    def run(self):
        """
        Run each step of the testcase
        """
        config = self.config
        for resolution in self.resolutions:
            ntasks = config.getint(f'planar_convergence',
                                   f'{resolution}km_ntasks')
            min_tasks = config.getint(f'planar_convergence',
                                      f'{resolution}km_min_tasks')
            step = self.steps['{}km_forward'.format(resolution)]
            step.ntasks = ntasks
            step.min_tasks = min_tasks

        # run the step
        super().run()

    def update_cores(self):
        """ Update the number of cores and min_tasks for each forward step """

        config = self.config

        goal_cells_per_core = config.getfloat('planar_convergence',
                                              'goal_cells_per_core')
        max_cells_per_core = config.getfloat('planar_convergence',
                                             'max_cells_per_core')

        section = config['planar_convergence']
        nx_1km = section.getint('nx_1km')
        ny_1km = section.getint('ny_1km')

        for resolution in self.resolutions:
            nx = int(nx_1km / resolution)
            ny = int(ny_1km / resolution)
            # a heuristic based on
            cell_count = nx*ny
            # ideally, about 300 cells per core
            # (make it a multiple of 4 because...it looks better?)
            ntasks = max(1, 4*round(cell_count / (4 * goal_cells_per_core)))
            # In a pinch, about 3000 cells per core
            min_tasks = max(1, round(cell_count / max_cells_per_core))
            step = self.steps[f'{resolution}km_forward']
            step.ntasks = ntasks
            step.min_tasks = min_tasks

            config.set('planar_convergence', f'{resolution}km_ntasks',
                       str(ntasks))
            config.set('planar_convergence', f'{resolution}km_min_tasks',
                       str(min_tasks))

    def _setup_steps(self, config):
        """
        setup steps given resolutions

        Parameters
        ----------
        config : compass.config.CompassConfigParser
            The config options containing the resolutions
        """

        resolutions = config.getlist('planar_convergence', 'resolutions',
                                     dtype=int)

        if self.resolutions is not None and self.resolutions == resolutions:
            return

        # start fresh with no steps
        self.steps = dict()
        self.steps_to_run = list()

        self.resolutions = resolutions

        for resolution in resolutions:
            self.add_step(self.create_init(resolution=resolution))
            self.add_step(Forward(test_case=self, resolution=resolution))

        self.add_step(self.create_analysis(resolutions=resolutions))

    def create_init(self, resolution):
        """

        Child class must override this to return an instance of a
        ConvergenceInit step

        Parameters
        ----------
        resolution : int
            The resolution of the step

        Returns
        -------
        init : compass.ocean.tests.planar_convergence.convergence_init.ConvergenceInit
            The init step object
        """

        pass

    def create_analysis(self, resolutions):
        """

        Child class must override this to return an instance of a
        ConvergenceInit step

        Parameters
        ----------
        resolutions : list of int
            The resolutions of the other steps in the test case

        Returns
        -------
        analysis : compass.ocean.tests.planar_convergence.conv_analysis.ConvAnalysis
            The init step object
        """

        pass
