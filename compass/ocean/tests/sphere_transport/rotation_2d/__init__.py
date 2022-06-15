from compass.testcase import TestCase

from compass.ocean.tests.sphere_transport.rotation_2d.mesh import Mesh
from compass.ocean.tests.sphere_transport.rotation_2d.init import Init
from compass.ocean.tests.sphere_transport.rotation_2d.forward import Forward
from compass.ocean.tests.sphere_transport.rotation_2d.analysis import \
    Analysis


class Rotation2D(TestCase):
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
            The test group that this case belongs to
        """
        super().__init__(test_group=test_group, name='rotation_2d')
        self.resolutions = None

    def configure(self):
        """
        Set config options for the test case
        """
        config = self.config
        resolutions = config.getlist('rotation_2d', 'resolutions', dtype=int)
        dtmin = config.getlist('rotation_2d', 'timestep_minutes', dtype=int)

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
            ntasks = config.getint('rotation_2d',
                                   f'QU{resolution}_ntasks')
            min_tasks = config.getint('rotation_2d',
                                      f'QU{resolution}_min_tasks')
            step = self.steps[f'QU{resolution}_forward']
            step.ntasks = ntasks
            step.min_tasks = min_tasks

        # run the step
        super().run()

    def update_cores(self):
        """ Update the number of cores and min_tasks for each forward step """

        config = self.config

        goal_cells_per_core = config.getfloat('rotation_2d',
                                              'goal_cells_per_core')
        max_cells_per_core = config.getfloat('rotation_2d',
                                             'max_cells_per_core')

        for resolution in self.resolutions:
            # a heuristic based on QU30 (65275 cells) and QU240 (10383 cells)
            approx_cells = 6e8 / resolution**2
            # ideally, about 300 cells per core
            # (make it a multiple of 4 because...it looks better?)
            ntasks = max(1,
                         4 * round(approx_cells / (4 * goal_cells_per_core)))
            # In a pinch, about 3000 cells per core
            min_tasks = max(1,
                            round(approx_cells / max_cells_per_core))
            step = self.steps[f'QU{resolution}_forward']
            step.ntasks = ntasks
            step.min_tasks = min_tasks

            config.set('rotation_2d', f'QU{resolution}_ntasks',
                       str(ntasks))
            config.set('rotation_2d', f'QU{resolution}_min_tasks',
                       str(min_tasks))
