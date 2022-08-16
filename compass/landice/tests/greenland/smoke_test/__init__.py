from compass.testcase import TestCase
from compass.landice.tests.greenland.run_model import RunModel


class SmokeTest(TestCase):
    """
    The default test case for the Greenland test group simply downloads the
    mesh and initial condition, then performs a short forward run on 36 cores.
    """

    def __init__(self, test_group, velo_solver):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.greenland.Greenland
            The test group that this test case belongs to

        velo_solver : {'sia', 'FO'}
            The velocity solver to use for the test case

        """
        name = 'smoke_test'
        subdir = '{}_{}'.format(velo_solver.lower(), name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        ntasks = 36
        if velo_solver == 'sia':
            min_tasks = 1
        elif velo_solver == 'FO':
            min_tasks = 1
        else:
            raise ValueError('Unexpected velo_solver {}'.format(velo_solver))

        self.add_step(
            RunModel(test_case=self, velo_solver=velo_solver, ntasks=ntasks,
                     min_tasks=min_tasks, openmp_threads=1))

    # no configure() method is needed because we will use the default dome
    # config options

    # no run() method is needed because we're doing the default: running all
    # steps
