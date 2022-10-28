from compass.testcase import TestCase
from compass.landice.tests.ismip6_run_ais.set_up_experiment \
        import SetUpExperiment
from compass.validate import compare_variables


class Projection(TestCase):
    """
    A test case for performing forward MALI runs of ISMIP6 Antaractic setup
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
        name = 'ismip6AE'

        super().__init__(test_group=test_group, name=name,
                         subdir=name)

        experiments = ['hist', 'ctrlAE', 'expAE01', 'expAE02', 'expAE03', 'expAE04', 'expAE05', 'expAE06']
        for exp in experiments:
            name = f'{exp}'
            self.add_step(
               SetUpExperiment(test_case=self, name=name, subdir=name, exp=exp))



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
