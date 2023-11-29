from math import floor

from compass.ocean.tests.dam_break.forward import Forward
from compass.ocean.tests.dam_break.initial_state import InitialState
from compass.ocean.tests.dam_break.lts.lts_regions import LTSRegions
from compass.ocean.tests.dam_break.viz import Viz
from compass.testcase import TestCase
from compass.validate import compare_variables


class Ramp(TestCase):
    """
    The default dam_break test case

    Attributes
    ----------
    resolution : float
        The resolution of the test case in km

    """

    def __init__(self, test_group, resolution, use_lts):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.dam_break.DamBreak
            The test group that this test case belongs to

        resolution : float
            The resolution of the test case in m

        use_lts : bool
            Whether local time-stepping is used

        """
        if use_lts:
            name = 'ramp_lts'
        else:
            name = 'ramp'

        self.resolution = resolution
        if resolution < 1.:
            res_name = f'{int(resolution*1e3)}cm'
        else:
            res_name = f'{int(resolution)}m'
        min_tasks = int(40 / (resolution / 0.04)**2)
        ntasks = 10 * min_tasks
        subdir = f'{res_name}/{name}'
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        init_step = InitialState(test_case=self, use_lts=use_lts)
        self.add_step(init_step)

        if use_lts:
            self.add_step(LTSRegions(test_case=self, init_step=init_step))

        forward_step = Forward(test_case=self, resolution=resolution,
                               use_lts=use_lts,
                               ntasks=ntasks, min_tasks=min_tasks,
                               openmp_threads=1)
        forward_step.add_namelist_options({'config_zero_drying_velocity_ramp':
                                           ".true."})
        self.add_step(forward_step)
        self.add_step(Viz(test_case=self))

    def configure(self):
        """
        Modify the configuration options for this test case.
        """

        resolution = self.resolution
        config = self.config
        dc = resolution  # cell width in m
        dx = 13          # width of the domain in m
        dy = 28          # length of the domain in m
        nx = round(dx / dc)
        ny = int(2 * floor(dy / (2 * dc)))  # guarantee that ny is even

        config.set('dam_break', 'nx', f'{nx}', comment='the number of '
                   'mesh cells in the x direction')
        config.set('dam_break', 'ny', f'{ny}', comment='the number of '
                   'mesh cells in the y direction')
        config.set('dam_break', 'dc', f'{dc}', comment='the distance '
                   'between adjacent cell centers')

    def validate(self):
        """
        Validate variables against a baseline
        """
        variables = ['layerThickness', 'normalVelocity', 'ssh']
        compare_variables(test_case=self, variables=variables,
                          filename1='forward/output.nc')
