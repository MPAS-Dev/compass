from compass.validate import compare_variables
from compass.testcase import TestCase
from compass.landice.tests.greenland.run_model import RunModel


class DecompositionTest(TestCase):
    """
    A test case for performing two MALI runs of the Greenland Ice Sheet setup,
    one with one core and one with eight.  The test case verifies that the
    results of the two runs are identical.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.greenland.Greenland
            The test group that this test case belongs to
        """
        name = 'decomposition_test'
        super().__init__(test_group=test_group, name=name)

        for procs in [1, 8]:
            name = '{}proc_run'.format(procs)
            RunModel(test_case=self, name=name, subdir=name, cores=procs,
                     threads=1)

    # no configure() method is needed

    def run(self):
        """
        Run each step of the test case
        """
        # run the steps
        super().run()

        variables = ['thickness', 'normalVelocity']
        steps = self.steps_to_run
        if '1proc_run' in steps and '8proc_run' in steps:
            compare_variables(variables, self.config, work_dir=self.work_dir,
                              filename1='1proc_run/output.nc',
                              filename2='8proc_run/output.nc')
