from compass.testcase import TestCase
from compass.landice.tests.ismip6_run_ais import RunModel
from compass.validate import compare_variables


class Projection(TestCase):
    """
    Todo: fix the doc string
    Todo: humboldt doc string is wrong. should be fixed (trevor)
    A test case for performing two MALI runs of a humboldt setup, one with one
    core and one with four.  The test case verifies that the results of the
    two runs are identical or close to identical.  The FO velocity solver is
    not bit for bit across decompositions, so identical results are not
    expected when it is used.
    """

    def __init__(self, test_group, mesh_type):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ismip6_run_ais.Ismip6RunAIS
            The test group that this test case belongs to

        mesh_type : {'mid', 'high'}
            The resolution or type of mesh of the test case

        """
        name = 'projection'

        # build dir name.
        subdir = f'{test_group.meshdirs[mesh_type]}/{name}'

        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(RunModel(test_case=self, mesh_type=mesh_type))

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        var_list = ['thickness']
        filename = f'{self.steps["run_model"].subdir}/output.nc'
        compare_variables(test_case=self,
                          variables=var_list,
                          filename1=filename)
