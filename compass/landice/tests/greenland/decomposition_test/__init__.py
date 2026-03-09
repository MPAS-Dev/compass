from compass.landice.tests.greenland.run_model import RunModel
from compass.parallel import get_available_parallel_resources
from compass.testcase import TestCase
from compass.validate import compare_variables


class DecompositionTest(TestCase):
    """
    A test case for performing two MALI runs of the Greenland Ice Sheet setup
    with different decompositions. The larger decomposition targets up to 32
    tasks, subject to available resources, and the smaller decomposition is
    roughly half of the larger one.

    Attributes
    ----------
    velo_solver : str
        The velocity solver used for the test case

    proc_list : list of int
        The pair of processor counts used in the decomposition comparison

    run_dirs : list of str
        The names of the subdirectories for the two decomposition runs
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
        self.proc_list = None
        self.run_dirs = None
        subdir = '{}_{}'.format(velo_solver.lower(), name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

    def configure(self):
        """
        Choose decomposition sizes from framework-detected resources and add
        run steps.

        The larger decomposition targets up to 32 tasks. FO runs require at
        least 10 tasks; SIA runs require at least 2 tasks.
        """
        available_resources = get_available_parallel_resources(self.config)
        target_max_tasks = 32
        if self.velo_solver == 'FO':
            smallest_acceptable_max_tasks = 10
        elif self.velo_solver == 'sia':
            smallest_acceptable_max_tasks = 2
        else:
            raise ValueError(f'Unexpected velo_solver {self.velo_solver}')

        max_tasks = max(
            smallest_acceptable_max_tasks,
            min(target_max_tasks, available_resources['cores']))
        low_tasks = max(1, max_tasks // 2)
        self.proc_list = [low_tasks, max_tasks]

        self.run_dirs = []
        for procs in self.proc_list:
            name = '{}proc_run'.format(procs)
            if name in self.run_dirs:
                name = '{}_{}'.format(name, len(self.run_dirs) + 1)
            self.run_dirs.append(name)
            self.add_step(
                RunModel(test_case=self, velo_solver=self.velo_solver,
                         name=name,
                         subdir=name, ntasks=procs, min_tasks=procs,
                         openmp_threads=1))

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        name1 = self.run_dirs[0]
        name2 = self.run_dirs[1]
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
