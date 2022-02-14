from compass.validate import compare_variables
from compass.testcase import TestCase
from compass.landice.tests.humboldt.run_model import RunModel


class RestartTest(TestCase):
    """
    A test case for performing two MALI runs of a humboldt setup, one full
    run and one run broken into two segments with a restart.  The test case
    verifies that the results of the two runs are identical.

    Attributes
    ----------
    mesh_type : str
        The resolution or type of mesh of the test case

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
        name = 'restart_test'
        self.mesh_type = mesh_type
        self.velo_solver = velo_solver
        if calving_law:
           self.calving_law = calving_law
        else:
           self.calving_law = 'none'
        subdir = 'mesh-{}_restart_test/velo-{}_calving-{}'.format(
                 mesh_type, velo_solver.lower(), calving_law.lower(), name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        name = 'full_run'
        step = RunModel(test_case=self, name=name, subdir=name, cores=12,
                        threads=1, velo_solver=velo_solver,
                        calving_law=self.calving_law,
                        mesh_type=mesh_type)
        # modify the namelist options and streams file
        step.add_namelist_file(
            'compass.landice.tests.humboldt.restart_test',
            'namelist.full', out_name='namelist.landice')
        step.add_streams_file(
            'compass.landice.tests.humboldt.restart_test',
            'streams.full', out_name='streams.landice')
        self.add_step(step)

        name = 'restart_run'
        step = RunModel(test_case=self, name=name, subdir=name, cores=12,
                        threads=1, velo_solver=velo_solver,
                        calving_law=self.calving_law,
                        mesh_type=mesh_type,
                        suffixes=['landice', 'landice.rst'])

        # modify the namelist options and streams file
        step.add_namelist_file(
            'compass.landice.tests.humboldt.restart_test',
            'namelist.restart', out_name='namelist.landice')
        step.add_streams_file(
            'compass.landice.tests.humboldt.restart_test',
            'streams.restart', out_name='streams.landice')

        step.add_namelist_file(
            'compass.landice.tests.humboldt.restart_test',
            'namelist.restart.rst', out_name='namelist.landice.rst')
        step.add_streams_file(
            'compass.landice.tests.humboldt.restart_test',
            'streams.restart.rst', out_name='streams.landice.rst')
        self.add_step(step)

    # no configure() method is needed

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['thickness', 'normalVelocity']
        if self.calving_law != 'none':
            variables.append('calvingThickness')
        compare_variables(test_case=self, variables=variables,
                          filename1='full_run/output.nc',
                          filename2='restart_run/output.nc')
