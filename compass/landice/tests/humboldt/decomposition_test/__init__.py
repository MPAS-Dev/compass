from compass.validate import compare_variables
from compass.testcase import TestCase
# from compass.landice.tests.humboldt.mesh import Mesh
from compass.landice.tests.humboldt.run_model import RunModel


class DecompositionTest(TestCase):
    """
    A test case for performing two MALI runs of a humboldt setup, one with one
    core and one with four.  The test case verifies that the results of the
    two runs are identical.

    Attributes
    ----------
    mesh_type : str
        The resolution or type of mesh of the test case

    velo_solver : str
        The velocity solver used for the test case

    calving_law : str
        The calving law used for the test case

    proc_list : list
        The pair of processor count values to test over.
        Function of velocity solver.
    """

    def __init__(self, test_group, velo_solver, calving_law, mesh_type):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.humboldt
            The test group that this test case belongs to

        velo_solver : {'sia', 'FO'}
            The velocity solver to use for the test case

        calving_law : str
            The calving law used for the test case

        mesh_type : {'1km', '3km'}
            The resolution or type of mesh of the test case
        """
        name = 'decomposition_test'
        self.mesh_type = mesh_type
        self.velo_solver = velo_solver
        assert self.velo_solver in {'sia', 'FO', 'none'}, \
            "Value of velo_solver must be one of {'sia', 'FO', 'none'}"
        if calving_law:
            self.calving_law = calving_law
        else:
            self.calving_law = 'none'
        subdir = 'mesh-{}_decomposition_test/velo-{}_calving-{}'.format(
                 mesh_type, velo_solver.lower(), calving_law.lower())
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        # Commented code to make use of mesh generation step
        # Note it will not include uReconstructX/Y or muFriction!
        # It will also add a few minutes run time to the test!
        # self.add_step(Mesh(test_case=self))
        if self.velo_solver == 'FO':
            self.proc_list = [16, 32]
        else:
            self.proc_list = [1, 32]
        for procs in self.proc_list:
            name = '{}proc_run'.format(procs)
            self.add_step(
                RunModel(test_case=self, name=name, subdir=name, cores=procs,
                         threads=1, velo_solver=self.velo_solver,
                         calving_law=self.calving_law,
                         mesh_type=mesh_type))

    # no configure() method is needed

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        run_dir1 = '{}proc_run'.format(self.proc_list[0])
        run_dir2 = '{}proc_run'.format(self.proc_list[1])

        var_list = ['thickness', ]
        if self.velo_solver == 'sia':
            var_list.append('surfaceSpeed')
        if self.calving_law != 'none':
            var_list.append('calvingThickness')

        if self.velo_solver in {'sia', 'none'}:

            compare_variables(test_case=self,
                              variables=var_list,
                              filename1=run_dir1+'/output.nc',
                              filename2=run_dir2+'/output.nc')

        elif self.velo_solver == 'FO':
            # validate thickness
            variable = ['thickness', ]
            l1_norm = 1.0e-10
            l2_norm = 1.0e-11
            linf_norm = 1.0e-11
            compare_variables(test_case=self, variables=variable,
                              filename1=run_dir1+'/output.nc',
                              filename2=run_dir2+'/output.nc',
                              l1_norm=l1_norm, l2_norm=l2_norm,
                              linf_norm=linf_norm, quiet=False)

            # validate normalVelocity
            variable = ['surfaceSpeed', ]
            l1_norm = 1.0e-15
            l2_norm = 1.0e-16
            linf_norm = 1.0e-17
            compare_variables(test_case=self, variables=variable,
                              filename1=run_dir1+'/output.nc',
                              filename2=run_dir2+'/output.nc',
                              l1_norm=l1_norm, l2_norm=l2_norm,
                              linf_norm=linf_norm, quiet=False)

            if 'calvingThickness' in var_list:
                l1_norm = 1.0e-11
                l2_norm = 1.0e-11
                linf_norm = 1.0e-12
                compare_variables(test_case=self,
                                  variables=['calvingThickness' ,],
                                  filename1=run_dir1+'/output.nc',
                                  filename2=run_dir2+'/output.nc',
                                  l1_norm=l1_norm, l2_norm=l2_norm,
                                  linf_norm=linf_norm, quiet=False)
