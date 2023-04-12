from compass.ocean.tests.internal_wave.forward import Forward
from compass.ocean.tests.internal_wave.initial_state import InitialState
from compass.ocean.tests.internal_wave.viz import Viz
from compass.testcase import TestCase
from compass.validate import compare_variables


class Default(TestCase):
    """
    The default test case for the internal wave test
    """

    def __init__(self, test_group, vlr=False):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.internal_wave.InternalWave
            The test group that this test case belongs to

        vlr : bool, optional
            Whether vertical Lagrangian remapping will be tested
        """
        name = 'default'
        if vlr:
            subdir = f'vlr/{name}'
        else:
            subdir = name
        super().__init__(test_group=test_group, subdir=subdir, name=name)
        self.add_step(InitialState(test_case=self))
        self.add_step(Forward(test_case=self, ntasks=4, openmp_threads=1,
                              vlr=vlr))
        self.add_step(Viz(test_case=self), run_by_default=False)

    def validate(self):
        """
        Validate variables against a baseline
        """
        compare_variables(test_case=self,
                          variables=['layerThickness', 'normalVelocity'],
                          filename1='forward/output.nc')
