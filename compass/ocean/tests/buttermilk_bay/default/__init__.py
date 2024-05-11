import numpy as np

from compass.config import CompassConfigParser
from compass.ocean.tests.buttermilk_bay.forward import Forward
from compass.ocean.tests.buttermilk_bay.initial_state import InitialState
from compass.ocean.tests.buttermilk_bay.viz import Viz
from compass.testcase import TestCase
from compass.validate import compare_variables


class Default(TestCase):
    """
    The default buttermilk_bay test case

    Attributes
    ----------
    wetdry : str
        The type of wetting and drying (``standard``, ``subgrid``)
    """

    def __init__(self, test_group, wetdry):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.buttermilk_bay.ButtermilkBay
            The test group that this test case belongs to

        wetdry : str
            The type of wetting and drying used (``standard``, ``subgrid``)
        """
        name = wetdry
        subdir = wetdry
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.resolutions = None
        self.wetdry = wetdry
        # add the steps with default resolutions so they can be listed
        config = CompassConfigParser()
        config.add_from_package('compass.ocean.tests.buttermilk_bay',
                                'buttermilk_bay.cfg')
        self._setup_steps(config)

    def configure(self):
        """
        Set config options for the test case
        """
        config = self.config
        # set up the steps again in case a user has provided new resolutions
        self._setup_steps(config)

        self.update_cores()

    def update_cores(self):
        """ Update the number of cores and min_tasks for each forward step """

        config = self.config

        goal_cells_per_core = config.getfloat('buttermilk_bay',
                                              'goal_cells_per_core')
        max_cells_per_core = config.getfloat('buttermilk_bay',
                                             'max_cells_per_core')
        lx = config.getfloat('buttermilk_bay', 'Lx')
        ly = config.getfloat('buttermilk_bay', 'Ly')

        for resolution in self.resolutions:

            nx = 2 * int(0.5 * lx / resolution + 0.5)
            ny = 2 * int(0.5 * ly * (2. / np.sqrt(3)) / resolution + 0.5)

            approx_cells = nx * ny
            # ideally, about 300 cells per core
            # (make it a multiple of 4 because...it looks better?)
            ntasks = max(1,
                         4 * round(approx_cells / (4 * goal_cells_per_core)))
            # In a pinch, about 3000 cells per core
            min_tasks = max(1,
                            round(approx_cells / max_cells_per_core))

            res_name = f'{resolution}m'
            step = self.steps[f'forward_{res_name}']
            step.ntasks = ntasks
            step.min_tasks = min_tasks

            config.set('buttermilk_bay', f'{res_name}_ntasks', str(ntasks),
                       comment=f'Target core count for {res_name} mesh')
            config.set('buttermilk_bay', f'{res_name}_min_tasks',
                       str(min_tasks),
                       comment=f'Minimum core count for {res_name} mesh')

    def _setup_steps(self, config):
        """ setup steps given resolutions """

        default_resolutions = '256, 128, 64'

        # set the default values that a user may change before setup
        config.set('buttermilk_bay', 'resolutions', default_resolutions,
                   comment='a list of resolutions (m) to test')

        # get the resolutions back, perhaps with values set in the user's
        # config file
        resolutions = config.getlist('buttermilk_bay',
                                     'resolutions', dtype=int)

        if self.resolutions is not None and self.resolutions == resolutions:
            return

        # start fresh with no steps
        self.steps = dict()
        self.steps_to_run = list()

        self.resolutions = resolutions

        for resolution in self.resolutions:

            res_name = f'{resolution}m'

            init_step = InitialState(test_case=self,
                                     name=f'initial_state_{res_name}',
                                     resolution=resolution,
                                     wetdry=self.wetdry)
            self.add_step(init_step)
            self.add_step(Forward(test_case=self,
                                  name=f'forward_{res_name}',
                                  resolution=resolution,
                                  wetdry=self.wetdry))
        self.add_step(Viz(test_case=self,
                          wetdry=self.wetdry,
                          resolutions=resolutions))

    def validate(self):
        """
        Validate variables against a baseline
        """
        super().validate()
        variables = ['layerThickness', 'normalVelocity']
        for res in self.resolutions:
            compare_variables(test_case=self, variables=variables,
                              filename1=f'forward_{res}m/output.nc')
