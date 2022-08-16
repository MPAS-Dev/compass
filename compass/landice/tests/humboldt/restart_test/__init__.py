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

    damage : str
        The damage method used for the test case

    face_melt : bool
        Whether to include face melting
    """

    def __init__(self, test_group, velo_solver, calving_law, mesh_type,
                 damage=None, face_melt=False):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.humboldt.Humboldt
            The test group that this test case belongs to

        velo_solver : {'sia', 'FO'}
            The velocity solver to use for the test case

        calving_law : str
            The calving law used for the test case

        mesh_type : {'1km', '3km'}
            The resolution or type of mesh of the test case

        damage : str
            The damage method used for the test case

        face_melt : bool
            Whether to include face melting
        """
        name = 'restart_test'
        self.mesh_type = mesh_type
        self.velo_solver = velo_solver
        assert self.velo_solver in {'sia', 'FO', 'none'}, \
            "Value of velo_solver must be one of {'sia', 'FO', 'none'}"
        self.calving_law = calving_law
        self.damage = damage
        self.face_melt = face_melt

        # build dir name.  always include velo solver and calving law
        subdir = 'mesh-{}_restart_test/velo-{}_calving-{}'.format(
                 mesh_type, velo_solver.lower(), calving_law.lower())
        # append damage and facemelt if provided
        if damage is not None:
            subdir += '_damage-{}'.format(damage)
        if face_melt is True:
            subdir += '_faceMelting'
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        name = 'full_run'
        step = RunModel(test_case=self, name=name, subdir=name, ntasks=32,
                        openmp_threads=1, velo_solver=velo_solver,
                        calving_law=self.calving_law,
                        damage=self.damage,
                        face_melt=self.face_melt,
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
        step = RunModel(test_case=self, name=name, subdir=name, ntasks=32,
                        openmp_threads=1, velo_solver=velo_solver,
                        calving_law=self.calving_law,
                        damage=self.damage,
                        face_melt=self.face_melt,
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
        variables = ['thickness', 'surfaceSpeed']

        if self.calving_law is not None and self.calving_law != 'none':
            variables.extend(['calvingVelocity', 'calvingThickness'])

        if self.damage is not None and self.damage != 'none':
            variables.append('damage')

        if self.face_melt is True:
            variables.append('faceMeltingThickness')

        compare_variables(test_case=self, variables=variables,
                          filename1='full_run/output.nc',
                          filename2='restart_run/output.nc')
