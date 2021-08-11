import configparser

from compass.testcase import TestCase

from compass.ocean.tests.sphere_transport.divergent2D.mesh import Mesh
from compass.ocean.tests.sphere_transport.divergent2D.init import Init
from compass.ocean.tests.sphere_transport.divergent2D.forward import Forward
from compass.ocean.tests.sphere_transport.divergent2D.analysis import \
    Analysis


class Divergent2D(TestCase):
    """
    A test case for 2D transport on the sphere

    Attributes
    ----------
    resolutions : list of int
    """
    def __init__(self, test_group):
        """
        Create test case for creating a global MPAS-Ocean mesh

        Parameters
        ----------
        test_group : compass.ocean.tests.sphere_transport.SphereTransport
        """
        super().__init__(test_group=test_group, name='divergent2D')
        self.resolutions = None

    def configure(self):
        """
        Set config options for the test case
        """
        config = self.config
        resolutions = config.get('divergent2D', 'resolutions')
        resolutions = [int(resolution) for resolution in
                       resolutions.replace(',', ' ').split()]
        dtmin = config.get('divergent2D', 'timestep_minutes')
        dtmin = [int(dt) for dt in dtmin.replace(',', ' ').split()]

        self.resolutions = resolutions
        self.timesteps = dtmin

        for i, resolution in enumerate(resolutions):
            self.add_step(Mesh(test_case=self, resolution=resolution))

            step = Init(test_case=self, resolution=resolution)
            self.add_step(step)

            self.add_step(Forward(test_case=self, resolution=resolution,
             dt_minutes=self.timesteps[i]))

        self.add_step(Analysis(test_case=self, resolutions=resolutions))

        self.update_cores()

    def run(self):
        """
        Run each step of the testcase
        """
        config = self.config
        for resolution in self.resolutions:
            cores = config.getint('divergent2D',
                                  'QU{}_cores'.format(resolution))
            min_cores = config.getint('divergent2D',
                                      'QU{}_min_cores'.format(resolution))
            step = self.steps['QU{}_forward'.format(resolution)]
            step.cores = cores
            step.min_cores = min_cores

        # run the step
        super().run()

    def update_cores(self):
        """ Update the number of cores and min_cores for each forward step """

        config = self.config

        goal_cells_per_core = config.getfloat('divergent2D',
                                              'goal_cells_per_core')
        max_cells_per_core = config.getfloat('divergent2D',
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

            config.set('divergent2D', 'QU{}_cores'.format(resolution),
                       str(cores))
            config.set('divergent2D', 'QU{}_min_cores'.format(resolution),
                       str(min_cores))
