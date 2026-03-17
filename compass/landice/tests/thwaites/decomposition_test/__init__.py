from compass.landice.tests.thwaites.run_model import RunModel
from compass.testcase import TestCase
from compass.validate import compare_variables


class DecompositionTest(TestCase):
    """
    A test case for performing two MALI runs of the Thwaites setup,
    with two different core counts.  The test case verifies that the
    results of the two runs are identical.
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
        super().__init__(test_group=test_group, name=name)

        self.cores_set = [16, 32]

        for procs in self.cores_set:
            name = '{}proc_run'.format(procs)
            step = RunModel(test_case=self, name=name, 
                            depth_integrated=depth_integrated,
                            ntasks=procs, min_tasks=procs, openmp_threads=1)
            if advection_type == 'fct':
                step.add_namelist_options(
                    {'config_thickness_advection': "'fct'",
                     'config_tracer_advection': "'fct'"},
                    out_name='namelist.landice')
            self.add_step(step)
    # no configure() method is needed

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        name1 = '{}proc_run'.format(self.cores_set[0])
        name2 = '{}proc_run'.format(self.cores_set[1])
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
