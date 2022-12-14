from compass.ocean.tests.parabolic_bowl.forward import Forward
from compass.ocean.tests.parabolic_bowl.initial_state import InitialState
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

    def __init__(self, test_group, ramp_type, wetdry):
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
        """
        name = f'{wetdry}_{ramp_type}'

        subdir = f'{wetdry}/{ramp_type}'
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        resolutions = [5, 10, 20]
        for resolution in resolutions:

            res_name = f'{resolution}km'

            self.add_step(InitialState(test_case=self,
                                       name=f'initial_state_{res_name}',
                                       resolution=resolution,
                                       wetdry=wetdry))
            self.add_step(Forward(test_case=self,
                                  name=f'forward_{res_name}',
                                  resolution=resolution,
                                  ramp_type=ramp_type,
                                  wetdry=wetdry))
        self.add_step(Viz(test_case=self, resolutions=resolutions))

#    def validate(self):
#        """
#        Validate variables against a baseline
#        """
#        variables = ['layerThickness', 'normalVelocity']
#        compare_variables(test_case=self, variables=variables,
#                          filename1='forward/output.nc')
