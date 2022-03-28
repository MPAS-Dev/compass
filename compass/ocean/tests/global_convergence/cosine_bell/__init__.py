from compass.config import CompassConfigParser
from compass.testcase import TestCase

from compass.ocean.tests.global_convergence.cosine_bell.mesh import Mesh
from compass.ocean.tests.global_convergence.cosine_bell.init import Init
from compass.ocean.tests.global_convergence.cosine_bell.forward import Forward
from compass.ocean.tests.global_convergence.cosine_bell.analysis import \
    Analysis


class CosineBell(TestCase):
    """
    A test case for creating a global MPAS-Ocean mesh

    Attributes
    ----------
    resolutions : list of int
    """
    def __init__(self, test_group):
        """
        Create test case for creating a global MPAS-Ocean mesh

        Parameters
        ----------
        test_group : compass.ocean.tests.cosine_bell.GlobalOcean
            The global ocean test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='cosine_bell')
        self.resolutions = None

        # add the steps with default resolutions so they can be listed
        config = CompassConfigParser()
        config.add_from_package(self.__module__, '{}.cfg'.format(self.name))
        self._setup_steps(config)

    def configure(self):
        """
        Set config options for the test case
        """
        config = self.config
        # set up the steps again in case a user has provided new resolutions
        self._setup_steps(config)

        init_options = dict()
        for option in ['temperature', 'salinity', 'lat_center', 'lon_center',
                       'radius', 'psi0', 'vel_pd']:
            init_options['config_cosine_bell_{}'.format(option)] = \
                config.get('cosine_bell', option)

        for step in self.steps.values():
            if 'init' in step.name:
                step.add_namelist_options(options=init_options, mode='init')

        self.update_cores()

    def run(self):
        """
        Run each step of the testcase
        """
        config = self.config
        for resolution in self.resolutions:
            cores = config.getint('cosine_bell',
                                  'QU{}_cores'.format(resolution))
            min_cores = config.getint('cosine_bell',
                                      'QU{}_min_cores'.format(resolution))
            step = self.steps['QU{}_forward'.format(resolution)]
            step.cores = cores
            step.min_cores = min_cores

        # run the step
        super().run()

    def update_cores(self):
        """ Update the number of cores and min_cores for each forward step """

        config = self.config

        goal_cells_per_core = config.getfloat('cosine_bell',
                                              'goal_cells_per_core')
        max_cells_per_core = config.getfloat('cosine_bell',
                                             'max_cells_per_core')

        for resolution in self.resolutions:
            # a heuristic based on QU30 (65275 cells) and QU240 (10383 cells)
            approx_cells = 6e8 / resolution**2
            # ideally, about 300 cells per core
            # (make it a multiple of 4 because...it looks better?)
            cores = max(1,
                        4*round(approx_cells / (4 * goal_cells_per_core)))
            # In a pinch, about 3000 cells per core
            min_cores = max(1,
                            round(approx_cells / max_cells_per_core))
            step = self.steps['QU{}_forward'.format(resolution)]
            step.cores = cores
            step.min_cores = min_cores

            config.set('cosine_bell', 'QU{}_cores'.format(resolution),
                       str(cores))
            config.set('cosine_bell', 'QU{}_min_cores'.format(resolution),
                       str(min_cores))

    def _setup_steps(self, config):
        """ setup steps given resolutions """
        resolutions = config.get('cosine_bell', 'resolutions')
        resolutions = [int(resolution) for resolution in
                       resolutions.replace(',', ' ').split()]

        if self.resolutions is not None and self.resolutions == resolutions:
            return

        # start fresh with no steps
        self.steps = dict()
        self.steps_to_run = list()

        self.resolutions = resolutions

        for resolution in resolutions:
            self.add_step(Mesh(test_case=self, resolution=resolution))

            self.add_step(Init(test_case=self, resolution=resolution))

            self.add_step(Forward(test_case=self, resolution=resolution))

        self.add_step(Analysis(test_case=self, resolutions=resolutions))
