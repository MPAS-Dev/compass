from compass.validate import compare_variables
from compass.testcase import TestCase
from compass.landice.tests.hydro_radial.setup_mesh import SetupMesh
from compass.landice.tests.hydro_radial.run_model import RunModel
from compass.landice.tests.hydro_radial.visualize import Visualize


class DecompositionTest(TestCase):
    """
    A test case for performing two MALI runs of a radially symmetric
    hydrological setup, one with one core and one with three.  The test case
    verifies that the results of the two runs are identical.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.hydro_radial.HydroRadial
            The test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='decomposition_test')

        self.add_step(
            SetupMesh(test_case=self, initial_condition='zero'))

        for procs in [1, 3]:
            name = '{}proc_run'.format(procs)
            self.add_step(
                RunModel(test_case=self, name=name, subdir=name, ntasks=procs,
                         openmp_threads=1))

            input_dir = name
            name = 'visualize_{}'.format(name)
            step = Visualize(test_case=self, name=name, subdir=name,
                             input_dir=input_dir)
            self.add_step(step, run_by_default=False)

    # no configure() method is needed

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['waterThickness', 'waterPressure']
        compare_variables(test_case=self, variables=variables,
                          filename1='1proc_run/output.nc',
                          filename2='3proc_run/output.nc')
