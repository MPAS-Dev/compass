from compass.testcase import TestCase
from compass.ocean.tests.merry_go_round.initial_state import InitialState
from compass.ocean.tests.merry_go_round.forward import Forward
from compass.ocean.tests.merry_go_round.viz import Viz
#from compass.ocean.tests.merry_go_round.analysis import Analysis
from compass.ocean.tests import merry_go_round
from compass.validate import compare_variables

class ConvergenceTest(TestCase):
    """
    The convergence test case for the merry-go-round test
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.merry_go_round.MerryGoRound
            The test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='convergence_test')
        resolutions = ['1.25m', '2.5m', '5m']
        for resolution in resolutions:
            self.add_step(InitialState(test_case=self, resolution=resolution,
                                       name=f'initial_state_{resolution}'))
            self.add_step(Forward(test_case=self, resolution=resolution,
                                  ntasks=4, openmp_threads=1,
                                  name=f'forward_{resolution}'))
            self.add_step(Viz(test_case=self, resolution=resolution,
                              name=f'viz_{resolution}'))
        #self.add_step(Analysis(test_case=self, resolutions=resolutions))

    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        # TODO make this work for multiple resolutions
        #merry_go_round.configure(self.resolution, self.config)

    def validate(self):
        """
        Validate variables against a baseline
        """
        #TODO change to tracers
        #compare_variables(test_case=self,
        #                  variables=['layerThickness', 'normalVelocity'],
        #                  filename1='forward/output.nc')
