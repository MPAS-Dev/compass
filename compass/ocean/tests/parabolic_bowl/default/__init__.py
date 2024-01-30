import numpy as np

from compass.config import CompassConfigParser
from compass.ocean.tests.parabolic_bowl.forward import Forward
from compass.ocean.tests.parabolic_bowl.initial_state import InitialState
from compass.ocean.tests.parabolic_bowl.lts.lts_regions import LTSRegions
from compass.ocean.tests.parabolic_bowl.viz import Viz
from compass.testcase import TestCase
from compass.validate import compare_variables


class Default(TestCase):
    """
    The default parabolic_bowl test case

    Attributes
    ----------
    ramp_type : str
        The type of vertical coordinate (``ramp``, ``noramp``, etc.)
    """

    def __init__(self, test_group, ramp_type, wetdry, use_lts):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.parabolic_bowl.ParabolicBowl
            The test group that this test case belongs to

        ramp_type : str
            The type of vertical coordinate (``ramp``, ``noramp``)

        wetdry : str
            The type of wetting and drying used (``standard``, ``subgrid``)

        use_lts : bool
            Whether local time-stepping is used
        """
        if use_lts:
            name = f'{wetdry}_{ramp_type}_lts'
        else:
            name = f'{wetdry}_{ramp_type}'

        if use_lts:
            subdir = f'{wetdry}/{ramp_type}_lts'
        else:
            subdir = f'{wetdry}/{ramp_type}'
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.resolutions = None
        self.wetdry = wetdry
        self.ramp_type = ramp_type
        self.use_lts = use_lts

        # add the steps with default resolutions so they can be listed
        config = CompassConfigParser()
        config.add_from_package('compass.ocean.tests.parabolic_bowl',
                                'parabolic_bowl.cfg')
        self._setup_steps(config, use_lts)

    def configure(self):
        """
        Set config options for the test case
        """
        config = self.config
        use_lts = self.use_lts
        # set up the steps again in case a user has provided new resolutions
        self._setup_steps(config, use_lts)

        self.update_cores()

    def update_cores(self):
        """ Update the number of cores and min_tasks for each forward step """

        config = self.config

        goal_cells_per_core = config.getfloat('parabolic_bowl',
                                              'goal_cells_per_core')
        max_cells_per_core = config.getfloat('parabolic_bowl',
                                             'max_cells_per_core')

        for resolution in self.resolutions:

            lx = config.getfloat('parabolic_bowl', 'Lx')
            ly = config.getfloat('parabolic_bowl', 'Ly')

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

            res_name = f'{resolution}km'
            step = self.steps[f'forward_{res_name}']
            step.ntasks = ntasks
            step.min_tasks = min_tasks

            config.set('parabolic_bowl', f'{res_name}_ntasks', str(ntasks),
                       comment=f'Target core count for {res_name} mesh')
            config.set('parabolic_bowl', f'{res_name}_min_tasks',
                       str(min_tasks),
                       comment=f'Minimum core count for {res_name} mesh')

    def _setup_steps(self, config, use_lts):
        """ setup steps given resolutions """

        default_resolutions = '20, 10, 5'

        # set the default values that a user may change before setup
        config.set('parabolic_bowl', 'resolutions', default_resolutions,
                   comment='a list of resolutions (km) to test')

        # get the resolutions back, perhaps with values set in the user's
        # config file
        resolutions = config.getlist('parabolic_bowl',
                                     'resolutions', dtype=int)

        if self.resolutions is not None and self.resolutions == resolutions:
            return

        # start fresh with no steps
        self.steps = dict()
        self.steps_to_run = list()

        self.resolutions = resolutions

        for resolution in self.resolutions:

            res_name = f'{resolution}km'

            init_step = InitialState(test_case=self,
                                     name=f'initial_state_{res_name}',
                                     resolution=resolution,
                                     wetdry=self.wetdry)
            self.add_step(init_step)
            if use_lts:
                self.add_step(LTSRegions(test_case=self,
                                         init_step=init_step,
                                         name=f'lts_regions_{res_name}',
                                         subdir=f'lts_regions_{res_name}'))
            self.add_step(Forward(test_case=self,
                                  name=f'forward_{res_name}',
                                  use_lts=use_lts,
                                  resolution=resolution,
                                  ramp_type=self.ramp_type,
                                  wetdry=self.wetdry))
        self.add_step(Viz(test_case=self, resolutions=resolutions,
                          use_lts=use_lts))

    def validate(self):
        """
        Validate variables against a baseline
        """
        super().validate()
        variables = ['layerThickness', 'normalVelocity']
        for res in self.resolutions:
            compare_variables(test_case=self, variables=variables,
                              filename1=f'forward_{res}km/output.nc')
