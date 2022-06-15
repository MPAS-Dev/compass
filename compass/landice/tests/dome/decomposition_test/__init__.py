from compass.validate import compare_variables
from compass.testcase import TestCase
from compass.landice.tests.dome.setup_mesh import SetupMesh
from compass.landice.tests.dome.run_model import RunModel
from compass.landice.tests.dome.visualize import Visualize


class DecompositionTest(TestCase):
    """
    A test case for performing two MALI runs of a dome setup, one with one core
    and one with four.  The test case verifies that the results of the two runs
    are identical.

    Attributes
    ----------
    mesh_type : str
        The resolution or type of mesh of the test case
    """

    def __init__(self, test_group, velo_solver, mesh_type):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.dome.Dome
            The test group that this test case belongs to

        velo_solver : {'sia', 'FO'}
            The velocity solver to use for the test case

        mesh_type : str
            The resolution or type of mesh of the test case
        """
        name = 'decomposition_test'
        self.mesh_type = mesh_type
        self.velo_solver = velo_solver
        subdir = '{}/{}_{}'.format(mesh_type, velo_solver.lower(), name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(
            SetupMesh(test_case=self, mesh_type=mesh_type))

        for procs in [1, 4]:
            name = '{}proc_run'.format(procs)
            self.add_step(
                RunModel(test_case=self, name=name, subdir=name, ntasks=procs,
                         openmp_threads=1, velo_solver=velo_solver,
                         mesh_type=mesh_type))

            input_dir = name
            name = 'visualize_{}'.format(name)
            step = Visualize(test_case=self, mesh_type=mesh_type, name=name,
                             subdir=name, input_dir=input_dir)
            self.add_step(step, run_by_default=False)

    # no configure() method is needed

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        if self.velo_solver == 'sia':
            compare_variables(test_case=self,
                              variables=['thickness', 'normalVelocity'],
                              filename1='1proc_run/output.nc',
                              filename2='4proc_run/output.nc')

        elif self.velo_solver == 'FO':
            # validate thickness
            variable = ['thickness', ]
            if self.mesh_type == 'variable_resolution':
                l1_norm = 1.0e-11
                l2_norm = 1.0e-12
                linf_norm = 1.0e-12
            elif self.mesh_type == '2000m':
                l1_norm = 1.0e-9
                l2_norm = 1.0e-11
                linf_norm = 1.0e-11
            compare_variables(test_case=self, variables=variable,
                              filename1='1proc_run/output.nc',
                              filename2='4proc_run/output.nc',
                              l1_norm=l1_norm, l2_norm=l2_norm,
                              linf_norm=linf_norm, quiet=False)

            # validate normalVelocity
            variable = ['normalVelocity', ]
            if self.mesh_type == 'variable_resolution':
                l1_norm = 1.0e-17
                l2_norm = 1.0e-18
                linf_norm = 1.0e-19
            elif self.mesh_type == '2000m':
                l1_norm = 1.0e-15
                l2_norm = 1.0e-16
                linf_norm = 1.0e-18
            compare_variables(test_case=self, variables=variable,
                              filename1='1proc_run/output.nc',
                              filename2='4proc_run/output.nc',
                              l1_norm=l1_norm, l2_norm=l2_norm,
                              linf_norm=linf_norm, quiet=False)
