from compass.validate import compare_variables
from compass.testcase import TestCase
from compass.landice.tests.circular_shelf.setup_mesh import SetupMesh
from compass.landice.tests.circular_shelf.run_model import RunModel
from compass.landice.tests.circular_shelf.visualize import Visualize


class DecompositionTest(TestCase):
    """
    A test case for performing two MALI runs of a circular shelf setup, one
    with one core and one with four.  The test case verifies that the results
    of the two runs are identical.

    Attributes
    ----------
    mesh_type : str
        The resolution or tye of mesh of the test case
    """

    def __init__(self, test_group, mesh_type):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.dome.Dome
            The test group that this test case belongs to

        mesh_type : str
            The resolution or tye of mesh of the test case
        """
        name = 'decomposition_test'
        self.mesh_type = mesh_type
        super().__init__(test_group=test_group, name=name)

        self.add_step(SetupMesh(test_case=self))

        for procs in [1, 4]:
            name = '{}proc_run'.format(procs)
            self.add_step(
                RunModel(test_case=self, name=name, subdir=name, ntasks=procs,
                         openmp_threads=1))

            input_dir = name
            name = 'visualize_{}'.format(name)
            step = Visualize(test_case=self, name=name,
                             subdir=name, input_dir=input_dir)
            self.add_step(step, run_by_default=False)

    # no configure() method is needed

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        # validate each variable with custom norm
        compare_variables(test_case=self, variables=['normalVelocity', ],
                          filename1='1proc_run/output.nc',
                          filename2='4proc_run/output.nc',
                          l1_norm=1.0e-3, l2_norm=1.0e-5,
                          linf_norm=1.0e-7, quiet=False)
        compare_variables(test_case=self, variables=['uReconstructX', ],
                          filename1='1proc_run/output.nc',
                          filename2='4proc_run/output.nc',
                          l1_norm=1.0e-4, l2_norm=2.0e-6,
                          linf_norm=1.0e-7, quiet=False)
        compare_variables(test_case=self, variables=['uReconstructY', ],
                          filename1='1proc_run/output.nc',
                          filename2='4proc_run/output.nc',
                          l1_norm=1.0e-4, l2_norm=2.0e-6,
                          linf_norm=1.0e-7, quiet=False)
