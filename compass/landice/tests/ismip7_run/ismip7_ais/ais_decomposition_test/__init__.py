from compass.landice.tests.ismip7_run.ismip7_ais.run_model import RunModel
from compass.landice.util import calculate_decomp_core_pair
from compass.testcase import TestCase
from compass.validate import compare_variables


class AisDecompositionTest(TestCase):
    """
    A test case for performing two short MALI runs of the ISMIP7 AIS
    configuration with different decompositions. The larger
    decomposition targets 128 tasks, subject to available resources,
    and the smaller decomposition is roughly half of the larger one.
    The test case verifies that results are identical (or very close
    to identical, given that the FO velocity solver used by this
    configuration is not guaranteed to be bit-for-bit across
    decompositions).

    Attributes
    ----------
    proc_list : list of int
        The pair of processor counts used in the decomposition
        comparison

    run_dirs : list of str
        The names of the subdirectories for the two decomposition runs
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ismip7_run.Ismip7Run
            The test group that this test case belongs to
        """
        name = 'ais_decomposition_test'
        super().__init__(test_group=test_group, name=name, subdir=name)
        self.proc_list = None
        self.run_dirs = None

    def configure(self):
        """
        Choose decomposition sizes from framework-detected resources and
        add run steps.

        The larger decomposition targets up to 128 tasks (the number of
        tasks used for full ISMIP7 AIS production runs). The FO velocity
        solver used by this configuration requires at least 10 tasks.
        """
        target_max_tasks = 128
        smallest_acceptable_max_tasks = 10

        self.proc_list = calculate_decomp_core_pair(
            self.config, target_max_tasks, smallest_acceptable_max_tasks)
        # Note: Failing when this many tasks are unavailable is
        # desired behavior for decomposition testing.

        self.run_dirs = []
        for procs in self.proc_list:
            name = f'{procs}proc_run'
            if name in self.run_dirs:
                name = f'{name}_{len(self.run_dirs) + 1}'
            self.run_dirs.append(name)
            self.add_step(
                RunModel(test_case=self, name=name, subdir=name,
                         ntasks=procs, min_tasks=procs))

    # no run() method is needed

    def validate(self):
        """
        Compare the results of the two decompositions
        """
        run_dir1 = self.run_dirs[0]
        run_dir2 = self.run_dirs[1]

        variables = ['thickness', 'surfaceSpeed', 'calvingVelocity',
                     'calvingThickness', 'floatingBasalMassBal']

        l1_norm = 0.0
        l2_norm = 0.0
        linf_norm = 0.0
        compare_variables(
            test_case=self, variables=variables,
            filename1=f'{run_dir1}/output/output_2d_2000.nc',
            filename2=f'{run_dir2}/output/output_2d_2000.nc',
            l1_norm=l1_norm, l2_norm=l2_norm, linf_norm=linf_norm,
            quiet=False)
