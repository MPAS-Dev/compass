from compass.testcase import TestCase
from compass.ocean.tests.merry_go_round.initial_state import InitialState
from compass.ocean.tests.merry_go_round.forward import Forward
from compass.ocean.tests.merry_go_round.viz import Viz
from compass.ocean.tests import merry_go_round
from compass.validate import compare_variables

class Default(TestCase):
    """
    The default test case for the merry-go-round test
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.merry_go_round.MerryGoRound
            The test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='default')
        self.resolution = '5m'
        self.add_step(InitialState(test_case=self, resolution=self.resolution,
                                   name=f'initial_state_{self.resolution}'))
        self.add_step(Forward(test_case=self, resolution=self.resolution,
                              ntasks=4, openmp_threads=1,
                              name=f'forward_{self.resolution}'))
        self.add_step(Viz(test_case=self, resolution=self.resolution,
                          name=f'viz_{self.resolution}'))

    def validate(self):
        """
        Validate variables against a baseline
        """
        compare_variables(test_case=self,
                          variables=['layerThickness', 'normalVelocity', 'tracer1'],
                          filename1=f'forward_{self.resolution}/output.nc')
