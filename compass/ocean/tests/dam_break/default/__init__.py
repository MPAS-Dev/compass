from compass.testcase import TestCase
from compass.ocean.tests.dam_break.initial_state import InitialState
from compass.ocean.tests.dam_break.forward import Forward


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
            The resolution of the test case in km

        """
        name = 'default'

        self.resolution = resolution
        if resolution < 1.:
            res_name = f'{int(resolution*1e3)}m'
        else:
            res_name = f'{int(resolution)}km'
        subdir = f'{res_name}/{name}'
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(InitialState(test_case=self))
        self.add_step(Forward(test_case=self, resolution=resolution,
                              cores=4, threads=1))

