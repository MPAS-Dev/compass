from math import floor
from compass.testcase import TestCase
from compass.ocean.tests.dam_break.initial_state import InitialState
from compass.ocean.tests.dam_break.forward import Forward
from compass.ocean.tests.dam_break.viz import Viz
from compass.validate import compare_variables


class Default(TestCase):
    """
    The default dam_break test case

    Attributes
    ----------
    resolution : float
        The resolution of the test case in km

    """

    def __init__(self, test_group, resolution):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.dam_break.DamBreak
            The test group that this test case belongs to

        resolution : float
            The resolution of the test case in m

        """
        name = 'default'

        self.resolution = resolution
        if resolution < 1.:
            res_name = f'{int(resolution*1e3)}cm'
        else:
            res_name = f'{int(resolution)}m'
        min_tasks = int(40/(resolution/0.04)**2)
        ntasks = 10*min_tasks
        subdir = f'{res_name}/{name}'
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(InitialState(test_case=self))
        self.add_step(Forward(test_case=self, resolution=resolution,
                              ntasks=ntasks, min_tasks=min_tasks,
                              openmp_threads=1))
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
        nx = round(dx/dc)
        ny = int(2*floor(dy/(2*dc)))  # guarantee that ny is even

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
                          filename1=f'forward/output.nc')
