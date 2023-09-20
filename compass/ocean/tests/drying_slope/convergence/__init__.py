from compass.config import CompassConfigParser
from compass.ocean.tests.drying_slope.analysis import Analysis
from compass.ocean.tests.drying_slope.forward import Forward
from compass.ocean.tests.drying_slope.initial_state import InitialState
from compass.ocean.tests.drying_slope.viz import Viz
from compass.testcase import TestCase
from compass.validate import compare_variables


class Convergence(TestCase):
    """
    The default drying_slope test case

    Attributes
    ----------
    resolution : float
        The resolution of the test case in km

    coord_type : str
        The type of vertical coordinate (``sigma``, ``single_layer``, etc.)
    """

    def __init__(self, test_group, coord_type):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.drying_slope.DryingSlope
            The test group that this test case belongs to

        coord_type : str
            The type of vertical coordinate (``sigma``, ``single_layer``)
        """
        name = 'convergence'

        self.coord_type = coord_type
        damping_coeffs = [0.01]
        self.damping_coeffs = damping_coeffs
        subdir = f'{coord_type}/{name}'
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)
        self.resolutions = None
        # add the steps with default resolutions so they can be listed
        config = CompassConfigParser()
        config.add_from_package('compass.ocean.tests.parabolic_bowl',
                                'parabolic_bowl.cfg')
        self._setup_steps(config)

    def _setup_steps(self, config):
        """ setup steps given resolutions """

        default_resolutions = '0.5, 1, 2'

        # set the default values that a user may change before setup
        config.set('drying_slope_convergence', 'resolutions',
                   default_resolutions,
                   comment='a list of resolutions (km) to test')

        # get the resolutions back, perhaps with values set in the user's
        # config file
        resolutions = config.getlist('drying_slope_convergence',
                                     'resolutions', dtype=float)

        if self.resolutions is not None and self.resolutions == resolutions:
            return

        # start fresh with no steps
        self.steps = dict()
        self.steps_to_run = list()

        self.resolutions = resolutions

        for resolution in self.resolutions:

            if resolution < 1.:
                res_name = f'{int(resolution*1e3)}m'
            else:
                res_name = f'{int(resolution)}km'
            self.add_step(InitialState(test_case=self,
                                       name=f'initial_state_{res_name}',
                                       resolution=resolution,
                                       coord_type=self.coord_type))
            self.add_step(Forward(test_case=self, resolution=resolution,
                                  name=f'forward_{res_name}',
                                  ntasks=4, openmp_threads=1,
                                  damping_coeff=self.damping_coeffs[0],
                                  coord_type=self.coord_type))
        self.add_step(Analysis(test_case=self,
                               resolutions=resolutions,
                               damping_coeff=self.damping_coeffs[0]))

    def validate(self):
        """
        Validate variables against a baseline
        """
        super().validate()
        variables = ['layerThickness', 'normalVelocity']
        damping_coeff = self.damping_coeffs[0]
        for resolution in self.resolutions:
            if resolution < 1.:
                res_name = f'{int(resolution*1e3)}m'
            else:
                res_name = f'{int(resolution)}km'
            name = f'{res_name}_{damping_coeff}'
            compare_variables(test_case=self, variables=variables,
                              filename1=f'forward_{name}/output.nc')
