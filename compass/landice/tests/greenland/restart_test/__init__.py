from compass.validate import compare_variables
from compass.testcase import TestCase
from compass.landice.tests.greenland.run_model import RunModel


class RestartTest(TestCase):
    """
    A test case for performing two MALI runs of the Greenland Ice Sheet setup,
    one full run and one run broken into two segments with a restart.  The
    test case verifies that the results of the two runs are identical.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.greenland.Greenland
            The test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='restart_test')

        name = 'full_run'
        step = RunModel(test_case=self, name=name, subdir=name, cores=4,
                        threads=1)
        # modify the namelist options and streams file
        step.add_namelist_file(
            'compass.landice.tests.greenland.restart_test',
            'namelist.full', out_name='namelist.landice')
        step.add_streams_file(
            'compass.landice.tests.greenland.restart_test',
            'streams.full', out_name='streams.landice')

        name = 'restart_run'
        step = RunModel(test_case=self, name=name, subdir=name, cores=4,
                        threads=1, suffixes=['landice', 'landice.rst'])

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

    # no configure() method is needed

    def run(self):
        """
        Run each step of the test case
        """
        # run the steps
        super().run()

        variables = ['thickness', 'normalVelocity']
        steps = self.steps_to_run
        if 'full_run' in steps and 'restart_run' in steps:
            compare_variables(variables, self.config, work_dir=self.work_dir,
                              filename1='full_run/output.nc',
                              filename2='restart_run/output.nc')
