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
        subdir = '{}/{}'.format(mesh_type, name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(
            SetupMesh(test_case=self, mesh_type=mesh_type))

        for procs in [1, 4]:
            name = '{}proc_run'.format(procs)
            self.add_step(
                RunModel(test_case=self, name=name, subdir=name, cores=procs,
                         threads=1, mesh_type=mesh_type))

            input_dir = name
            name = 'visualize_{}'.format(name)
            step = Visualize(test_case=self, mesh_type=mesh_type, name=name,
                             subdir=name, input_dir=input_dir)
            self.add_step(step, run_by_default=False)

    # no configure() method is needed

    def run(self):
        """
        Run each step of the test case
        """
        # run the steps
        super().run()

        variables = ['thickness', 'normalVelocity']
        steps = self.steps_to_run
        if '1proc_run' in steps and '4proc_run' in steps:
            compare_variables(variables, self.config, work_dir=self.work_dir,
                              filename1='1proc_run/output.nc',
                              filename2='4proc_run/output.nc')
