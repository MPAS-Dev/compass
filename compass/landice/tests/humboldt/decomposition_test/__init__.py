from compass.validate import compare_variables
from compass.testcase import TestCase
#from compass.landice.tests.humboldt.mesh import Mesh
from compass.landice.tests.humboldt.run_model import RunModel


class DecompositionTest(TestCase):
    """
    A test case for performing two MALI runs of a humboldt setup, one with one core
    and one with four.  The test case verifies that the results of the two runs
    are identical.

    Attributes
    ----------
    mesh_type : str
        The resolution or type of mesh of the test case
        
    velo_solver : str
        The velocity solver used for the test case

    calving_law : str
        The calving law used for the test case

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

        mesh_type : str
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
                 mesh_type, velo_solver.lower(), calving_law.lower(), name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        # Commented code to make use of mesh generation step
        # Note it will not include uReconstructX/Y or muFriction!
        # It will also add a few minutes run time to the test!
        #self.add_step(Mesh(test_case=self))

        for procs in [1, 32]:
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
        if self.velo_solver in {'sia', 'none'}:
            compare_variables(test_case=self,
                              variables=['thickness', 'normalVelocity'],
                              filename1='1proc_run/output.nc',
                              filename2='32proc_run/output.nc')

        elif self.velo_solver == 'FO':
            # validate thickness
            variable = ['thickness', ]
            l1_norm = 1.0e-11
            l2_norm = 1.0e-12
            linf_norm = 1.0e-12
            compare_variables(test_case=self, variables=variable,
                              filename1='1proc_run/output.nc',
                              filename2='32proc_run/output.nc',
                              l1_norm=l1_norm, l2_norm=l2_norm,
                              linf_norm=linf_norm, quiet=False)

            # validate normalVelocity
            variable = ['normalVelocity', ]
            l1_norm = 1.0e-17
            l2_norm = 1.0e-18
            linf_norm = 1.0e-19
            compare_variables(test_case=self, variables=variable,
                              filename1='1proc_run/output.nc',
                              filename2='32proc_run/output.nc',
                              l1_norm=l1_norm, l2_norm=l2_norm,
                              linf_norm=linf_norm, quiet=False)
