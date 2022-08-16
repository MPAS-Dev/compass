from compass.validate import compare_variables
from compass.testcase import TestCase
from compass.landice.tests.greenland.run_model import RunModel


class RestartTest(TestCase):
    """
    A test case for performing two MALI runs of the Greenland Ice Sheet setup,
    one full run and one run broken into two segments with a restart.  The
    test case verifies that the results of the two runs are identical.
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
        name = 'restart_test'
        subdir = '{}_{}'.format(velo_solver.lower(), name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        ntasks = 36
        if velo_solver == 'sia':
            min_tasks = 1
        elif velo_solver == 'FO':
            min_tasks = 4
        else:
            raise ValueError('Unexpected velo_solver {}'.format(velo_solver))

        name = 'full_run'
        step = RunModel(test_case=self, velo_solver=velo_solver, name=name,
                        subdir=name, ntasks=ntasks, min_tasks=min_tasks,
                        openmp_threads=1)
        # modify the namelist options and streams file
        step.add_namelist_file(
            'compass.landice.tests.greenland.restart_test',
            'namelist.full', out_name='namelist.landice')
        step.add_streams_file(
            'compass.landice.tests.greenland.restart_test',
            'streams.full', out_name='streams.landice')
        self.add_step(step)

        name = 'restart_run'
        step = RunModel(test_case=self, velo_solver=velo_solver, name=name,
                        subdir=name, ntasks=ntasks, min_tasks=min_tasks,
                        openmp_threads=1, suffixes=['landice', 'landice.rst'])

        # modify the namelist options and streams file
        step.add_namelist_file(
            'compass.landice.tests.greenland.restart_test',
            'namelist.restart', out_name='namelist.landice')
        step.add_streams_file(
            'compass.landice.tests.greenland.restart_test',
            'streams.restart', out_name='streams.landice')

        step.add_namelist_file(
            'compass.landice.tests.greenland.restart_test',
            'namelist.restart.rst', out_name='namelist.landice.rst')
        # same streams file for both restart stages
        step.add_streams_file(
            'compass.landice.tests.greenland.restart_test',
            'streams.restart', out_name='streams.landice.rst')
        self.add_step(step)

    # no configure() method is needed

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['thickness', 'normalVelocity']
        compare_variables(test_case=self, variables=variables,
                          filename1='full_run/output.nc',
                          filename2='restart_run/output.nc')
