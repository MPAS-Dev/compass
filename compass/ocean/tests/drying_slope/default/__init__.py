from compass.testcase import TestCase
from compass.ocean.tests.drying_slope.initial_state import InitialState
from compass.ocean.tests.drying_slope.forward import Forward
from compass.ocean.tests.drying_slope.viz import Viz


class Default(TestCase):
    """
    The default drying_slope test case

    Attributes
    ----------
    resolution : float
        The resolution of the test case in km

    coord_type : str
        The type of vertical coordinate (``sigma``, ``single_layer``, etc.)
    """

    def __init__(self, test_group, resolution, coord_type):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.drying_slope.DryingSlope
            The test group that this test case belongs to

        resolution : float
            The resolution of the test case in km

        coord_type : str
            The type of vertical coordinate (``sigma``, ``single_layer``)
        """
        name = 'default'

        self.resolution = resolution
        self.coord_type = coord_type
        if resolution < 1.:
            res_name = f'{int(resolution*1e3)}m'
        else:
            res_name = f'{int(resolution)}km'
        subdir = f'{res_name}/{coord_type}/{name}'
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(InitialState(test_case=self))
        for damping_coeff in [0.0025, 0.01]:
            self.add_step(Forward(test_case=self, resolution=resolution,
                                  cores=4, threads=1,
                                  damping_coeff=damping_coeff))
        self.add_step(Viz(test_case=self))

    def configure(self):
        """
        Modify the configuration options for this test case.
        """

        resolution = self.resolution
        config = self.config
        ny = round(28 / resolution)
        if resolution < 1.:
            ny += 2
        dc = 1e3 * resolution

        config.set('drying_slope', 'ny', f'{ny}', comment='the number of '
                   'mesh cells in the y direction')
        config.set('drying_slope', 'dc', f'{dc}', comment='the distance '
                   'between adjacent cell centers')
