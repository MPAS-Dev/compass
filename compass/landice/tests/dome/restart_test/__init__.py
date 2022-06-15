from compass.validate import compare_variables
from compass.testcase import TestCase
from compass.landice.tests.dome.setup_mesh import SetupMesh
from compass.landice.tests.dome.run_model import RunModel
from compass.landice.tests.dome.visualize import Visualize


class RestartTest(TestCase):
    """
    A test case for performing two MALI runs of a dome setup, one full run and
    one run broken into two segments with a restart.  The test case verifies
    that the results of the two runs are identical.

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
        name = 'restart_test'
        self.mesh_type = mesh_type
        self.velo_solver = velo_solver
        subdir = '{}/{}_{}'.format(mesh_type, velo_solver.lower(), name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(
            SetupMesh(test_case=self, mesh_type=mesh_type))

        name = 'full_run'
        step = RunModel(test_case=self, name=name, subdir=name, ntasks=4,
                        openmp_threads=1, velo_solver=velo_solver,
                        mesh_type=mesh_type)
        # modify the namelist options and streams file
        step.add_namelist_file(
            'compass.landice.tests.dome.restart_test',
            'namelist.full', out_name='namelist.landice')
        step.add_streams_file(
            'compass.landice.tests.dome.restart_test',
            'streams.full', out_name='streams.landice')
        self.add_step(step)

        input_dir = name
        name = 'visualize_{}'.format(name)
        step = Visualize(test_case=self, mesh_type=mesh_type, name=name,
                         subdir=name, input_dir=input_dir)
        self.add_step(step, run_by_default=False)

        name = 'restart_run'
        step = RunModel(test_case=self, name=name, subdir=name, ntasks=4,
                        openmp_threads=1, velo_solver=velo_solver,
                        mesh_type=mesh_type,
                        suffixes=['landice', 'landice.rst'])

        # modify the namelist options and streams file
        step.add_namelist_file(
            'compass.landice.tests.dome.restart_test',
            'namelist.restart', out_name='namelist.landice')
        step.add_streams_file(
            'compass.landice.tests.dome.restart_test',
            'streams.restart', out_name='streams.landice')

        step.add_namelist_file(
            'compass.landice.tests.dome.restart_test',
            'namelist.restart.rst', out_name='namelist.landice.rst')
        step.add_streams_file(
            'compass.landice.tests.dome.restart_test',
            'streams.restart.rst', out_name='streams.landice.rst')
        self.add_step(step)

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
        variables = ['thickness', 'normalVelocity']
        compare_variables(test_case=self, variables=variables,
                          filename1='full_run/output.nc',
                          filename2='restart_run/output.nc')
