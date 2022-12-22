from compass.testcase import TestCase
from compass.ocean.tests.turbulence_closure.initial_state import InitialState
from compass.ocean.tests.turbulence_closure.forward import Forward
from compass.ocean.tests.turbulence_closure.viz import Viz
from compass.ocean.tests import turbulence_closure


class Default(TestCase):
    """
    The default test case for the turbulence closure test group simply creates
    the mesh and initial condition, then performs a short forward run on 4
    cores.

    Attributes
    ----------
    resolution : str
        The resolution of the test case
    """

    def __init__(self, test_group, resolution, forcing='cooling'):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.turbulence_closure.TurbulenceClosure
            The test group that this test case belongs to

        resolution : str
            The resolution of the test case

        forcing: str
            The forcing applied to the test case
        """
        name = 'default'
        self.resolution = resolution
        self.forcing = forcing
        subdir = f'{resolution}/{forcing}/{name}'
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        if resolution == '1m' or resolution == '2m':
            ntasks = 128
            plot_comparison=True
        else:
            ntasks = 4
            plot_comparison=False

        self.add_step(
            InitialState(test_case=self, resolution=resolution))
        self.add_step(
            Forward(test_case=self, ntasks=ntasks, openmp_threads=1, resolution=resolution))
        self.add_step(Viz(test_case=self, resolution=resolution, forcing=forcing, do_comparison=plot_comparison))

    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        turbulence_closure.configure(self.resolution, self.forcing, self.config)

    # no run() is needed because we're doing the default: running all steps
