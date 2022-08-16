from compass.testcase import TestCase
from compass.ocean.tests.drying_slope.initial_state import InitialState
from compass.ocean.tests.drying_slope.forward import Forward
from compass.ocean.tests.drying_slope.viz import Viz
from compass.validate import compare_variables


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
        self.add_step(InitialState(test_case=self, coord_type=coord_type))
        if coord_type == 'single_layer':
            self.add_step(Forward(test_case=self, resolution=resolution,
                                  ntasks=4, openmp_threads=1,
                                  coord_type=coord_type))
            damping_coeffs = None
        else:
            damping_coeffs = [0.0025, 0.01]
            for damping_coeff in damping_coeffs:
                self.add_step(Forward(test_case=self, resolution=resolution,
                                      ntasks=4, openmp_threads=1,
                                      damping_coeff=damping_coeff,
                                      coord_type=coord_type))
        self.damping_coeffs = damping_coeffs
        self.add_step(Viz(test_case=self, damping_coeffs=damping_coeffs))

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

    def validate(self):
        """
        Validate variables against a baseline
        """
        damping_coeffs = self.damping_coeffs
        variables = ['layerThickness', 'normalVelocity']
        if damping_coeffs is not None:
            for damping_coeff in damping_coeffs:
                compare_variables(test_case=self, variables=variables,
                                  filename1=f'forward_{damping_coeff}/'
                                            'output.nc')
        else:
            compare_variables(test_case=self, variables=variables,
                              filename1='forward/output.nc')
