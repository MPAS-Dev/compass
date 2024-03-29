from compass.landice.tests.humboldt.run_model import RunModel
from compass.testcase import TestCase
from compass.validate import compare_variables


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

    depth_integrated  : bool
        Whether the (FO) velocity model is depth integrated

    hydro : bool
        Whether to include subglacial hydrology
    """

    def __init__(self, test_group, velo_solver, calving_law, mesh_type,
                 damage=None, face_melt=False, depth_integrated=False,
                 hydro=False):
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

        depth_integrated  : bool
            Whether the (FO) velocity model is depth integrated

        hydro : bool
            Whether to include subglacial hydrology
        """
        name = 'restart_test'
        self.mesh_type = mesh_type
        self.velo_solver = velo_solver
        assert self.velo_solver in {'sia', 'FO', 'none'}, \
            "Value of velo_solver must be one of {'sia', 'FO', 'none'}"
        self.calving_law = calving_law
        self.damage = damage
        self.face_melt = face_melt
        if hydro is not None:
            self.hydro = hydro
        else:
            self.hydro = False

        # build dir name.  always include velo solver and calving law
        subdir = 'mesh-{}_restart_test/velo-{}'.format(
                 mesh_type, velo_solver.lower())
        if velo_solver == 'FO' and depth_integrated is True:
            subdir += '-depthInt'
        subdir += '_calving-{}'.format(calving_law.lower())
        # append damage and facemelt if provided
        if damage is not None:
            subdir += '_damage-{}'.format(damage)
        if face_melt is True:
            subdir += '_faceMelting'
        if self.hydro is True:
            subdir += '_subglacialhydro'
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        name = 'full_run'
        step = RunModel(test_case=self, name=name, subdir=name, ntasks=32,
                        openmp_threads=1, velo_solver=velo_solver,
                        calving_law=self.calving_law,
                        damage=self.damage,
                        face_melt=self.face_melt,
                        depth_integrated=depth_integrated,
                        hydro=self.hydro,
                        mesh_type=mesh_type)
        # Hydro model restart tests should be shorter duration to keep the
        # tests to a reasonable runtime.  Different nl templates have been
        # set up for that
        if self.hydro is True:
            nl1 = 'namelist.full.hydro'
        else:
            nl1 = 'namelist.full'
        # modify the namelist options and streams file
        step.add_namelist_file(
            'compass.landice.tests.humboldt.restart_test',
            nl1, out_name='namelist.landice')
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
                        hydro=self.hydro,
                        mesh_type=mesh_type,
                        depth_integrated=depth_integrated,
                        suffixes=['landice', 'landice.rst'])

        # Hydro model restart tests should be shorter duration to keep the
        # tests to a reasonable runtime.  Different nl templates have been
        # set up for that
        if self.hydro is True:
            nl1 = 'namelist.restart.hydro'
            nl2 = 'namelist.restart.rst.hydro'
        else:
            nl1 = 'namelist.restart'
            nl2 = 'namelist.restart.rst'
        # modify the namelist options and streams file
        step.add_namelist_file(
            'compass.landice.tests.humboldt.restart_test',
            nl1, out_name='namelist.landice')
        step.add_streams_file(
            'compass.landice.tests.humboldt.restart_test',
            'streams.restart', out_name='streams.landice')

        step.add_namelist_file(
            'compass.landice.tests.humboldt.restart_test',
            nl2, out_name='namelist.landice.rst')
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

        if self.hydro is True:
            variables.extend(['waterThickness', 'hydropotential', 'waterFlux',
                              'channelDischarge', 'channelArea'])

        compare_variables(test_case=self, variables=variables,
                          filename1='full_run/output.nc',
                          filename2='restart_run/output.nc')
