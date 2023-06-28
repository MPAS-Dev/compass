from compass.landice.tests.thwaites.run_model import RunModel
from compass.landice.util import calculate_decomp_core_pair
from compass.testcase import TestCase
from compass.validate import compare_variables


class DecompositionTest(TestCase):
    """
    A test case for performing two MALI runs of the Thwaites setup with
    different decompositions. The larger decomposition targets up to 32
    tasks, subject to available resources, and the smaller decomposition is
    roughly half of the larger one. The test case verifies that the results
    of the two runs are identical.

    Attributes
    ----------
    depth_integrated : bool
        Whether the FO velocity model is depth integrated

    proc_list : list of int
        The pair of processor counts used in the decomposition comparison

    run_dirs : list of str
        The names of the subdirectories for the two decomposition runs
    """

    def __init__(self, test_group, advection_type, depth_integrated=False):
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
            name_tmp = 'fo-depthInt_decomposition_test'
        else:
            name_tmp = 'fo_decomposition_test'

        name = f'{advection_type}_{name_tmp}'
        self.advection_type = advection_type
        self.depth_integrated = depth_integrated
        self.proc_list = None
        self.run_dirs = None
        super().__init__(test_group=test_group, name=name)

    def configure(self):
        """
        Choose decomposition sizes from framework-detected resources and add
        run steps.

        The larger decomposition targets up to 32 tasks and requires at least
        10 tasks to run this decomposition test.
        """
        target_max_tasks = 32
        smallest_acceptable_max_tasks = 10
        self.proc_list = calculate_decomp_core_pair(
            self.config, target_max_tasks, smallest_acceptable_max_tasks)

        self.run_dirs = []
        for procs in self.proc_list:
            name = '{}proc_run'.format(procs)
            if name in self.run_dirs:
                name = '{}_{}'.format(name, len(self.run_dirs) + 1)
            self.run_dirs.append(name)
            step = RunModel(test_case=self, name=name,
                            depth_integrated=self.depth_integrated,
                            ntasks=procs, min_tasks=procs, openmp_threads=1)
            if self.advection_type == 'fct':
                step.add_namelist_options(
                    {'config_thickness_advection': "'fct'",
                     'config_tracer_advection': "'fct'"},
                    out_name='namelist.landice')
            self.add_step(step)

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        name1 = self.run_dirs[0]
        name2 = self.run_dirs[1]
        # validate thickness
        compare_variables(test_case=self,
                          variables=['thickness', ],
                          filename1='{}/output.nc'.format(name1),
                          filename2='{}/output.nc'.format(name2),
                          l1_norm=1.0e-11,
                          l2_norm=1.0e-11,
                          linf_norm=1.0e-12,
                          quiet=False)

        # validate surfaceSpeed
        compare_variables(test_case=self,
                          variables=['surfaceSpeed', ],
                          filename1='{}/output.nc'.format(name1),
                          filename2='{}/output.nc'.format(name2),
                          l1_norm=1.0e-13,
                          l2_norm=1.0e-14,
                          linf_norm=1.0e-15,
                          quiet=False)
