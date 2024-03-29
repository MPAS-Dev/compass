from compass.landice.tests.thwaites.run_model import RunModel
from compass.testcase import TestCase
from compass.validate import compare_variables


class RestartTest(TestCase):
    """
    A test case for performing two MALI runs of the Thwaites setup,
    one full run and one run broken into two segments with a restart.  The
    test case verifies that the results of the two runs are identical.
    """

    def __init__(self, test_group, depth_integrated=False):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.thwaites.Thwaites
            The test group that this test case belongs to

        depth_integrated  : bool
            Whether the (FO) velocity model is depth integrated

        """

        if depth_integrated is True:
            name = 'fo-depthInt_restart_test'
        else:
            name = 'fo_restart_test'

        super().__init__(test_group=test_group, name=name)

        ntasks = 36
        min_tasks = 4

        name = 'full_run'
        step = RunModel(test_case=self, name=name,
                        depth_integrated=depth_integrated,
                        ntasks=ntasks, min_tasks=min_tasks, openmp_threads=1)
        # modify the namelist options and streams file
        step.add_namelist_file(
            'compass.landice.tests.thwaites.restart_test',
            'namelist.full', out_name='namelist.landice')
        step.add_streams_file(
            'compass.landice.tests.thwaites.restart_test',
            'streams.full', out_name='streams.landice')
        self.add_step(step)

        name = 'restart_run'
        step = RunModel(test_case=self, name=name, ntasks=ntasks,
                        depth_integrated=depth_integrated,
                        min_tasks=min_tasks, openmp_threads=1,
                        suffixes=['landice', 'landice.rst'])

        # modify the namelist options and streams file
        step.add_namelist_file(
            'compass.landice.tests.thwaites.restart_test',
            'namelist.restart', out_name='namelist.landice')
        step.add_streams_file(
            'compass.landice.tests.thwaites.restart_test',
            'streams.restart', out_name='streams.landice')

        step.add_namelist_file(
            'compass.landice.tests.thwaites.restart_test',
            'namelist.restart.rst', out_name='namelist.landice.rst')
        # same streams file for both restart stages
        step.add_streams_file(
            'compass.landice.tests.thwaites.restart_test',
            'streams.restart', out_name='streams.landice.rst')
        self.add_step(step)

    # no configure() method is needed

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['thickness', 'surfaceSpeed']
        compare_variables(test_case=self, variables=variables,
                          filename1='full_run/output.nc',
                          filename2='restart_run/output.nc')
