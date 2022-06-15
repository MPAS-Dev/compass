from compass.validate import compare_variables
from compass.testcase import TestCase
from compass.landice.tests.greenland.run_model import RunModel


class DecompositionTest(TestCase):
    """
    A test case for performing two MALI runs of the Greenland Ice Sheet setup,
    one with one core and one with eight.  The test case verifies that the
    results of the two runs are identical.
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
        name = 'decomposition_test'
        self.velo_solver = velo_solver
        subdir = '{}_{}'.format(velo_solver.lower(), name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        if velo_solver == 'sia':
            self.cores_set = [1, 8]
        elif velo_solver == 'FO':
            self.cores_set = [16, 32]
        else:
            raise ValueError('Unexpected velo_solver {}'.format(velo_solver))

        for procs in self.cores_set:
            name = '{}proc_run'.format(procs)
            self.add_step(
                RunModel(test_case=self, velo_solver=velo_solver, name=name,
                         subdir=name, ntasks=procs, min_tasks=procs,
                         openmp_threads=1))

    # no configure() method is needed

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        name1 = '{}proc_run'.format(self.cores_set[0])
        name2 = '{}proc_run'.format(self.cores_set[1])
        if self.velo_solver == 'sia':
            compare_variables(test_case=self,
                              variables=['thickness', 'normalVelocity'],
                              filename1='{}/output.nc'.format(name1),
                              filename2='{}/output.nc'.format(name2))

        elif self.velo_solver == 'FO':
            # validate thickness
            compare_variables(test_case=self,
                              variables=['thickness', ],
                              filename1='{}/output.nc'.format(name1),
                              filename2='{}/output.nc'.format(name2),
                              l1_norm=1.0e-4,
                              l2_norm=1.0e-4,
                              linf_norm=1.0e-4,
                              quiet=False)

            # validate normalVelocity
            compare_variables(test_case=self,
                              variables=['normalVelocity', ],
                              filename1='{}/output.nc'.format(name1),
                              filename2='{}/output.nc'.format(name2),
                              l1_norm=1.0e-5,
                              l2_norm=1.0e-5,
                              linf_norm=1.0e-5,
                              quiet=False)
