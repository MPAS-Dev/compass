from compass.landice.tests.humboldt.run_model import RunModel
from compass.testcase import TestCase
from compass.validate import compare_variables


class DecompositionTest(TestCase):
    """
    A test case for performing two MALI runs of a humboldt setup, one with one
    core and one with four.  The test case verifies that the results of the
    two runs are identical or close to identical.  The FO velocity solver is
    not bit for bit across decompositions, so identical results are not
    expected when it is used.

    Attributes
    ----------
    mesh_type : str
        The resolution or type of mesh of the test case

    velo_solver : str
        The velocity solver used for the test case

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

    proc_list : list
        The pair of processor count values to test over.
        Function of velocity solver.
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
        name = 'decomposition_test'
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
        subdir = 'mesh-{}_decomposition_test/velo-{}'.format(
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

        if self.velo_solver == 'FO':
            self.proc_list = [16, 32]
        else:
            self.proc_list = [1, 32]
        for procs in self.proc_list:
            name = '{}proc_run'.format(procs)
            self.add_step(
                RunModel(test_case=self, name=name, subdir=name, ntasks=procs,
                         openmp_threads=1, velo_solver=self.velo_solver,
                         calving_law=self.calving_law,
                         damage=self.damage,
                         face_melt=self.face_melt,
                         depth_integrated=depth_integrated,
                         hydro=self.hydro,
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

        var_list = ['thickness']
        if self.velo_solver == 'sia':
            var_list.append('surfaceSpeed')

        if self.calving_law is not None and self.calving_law != 'none':
            var_list.extend(['calvingVelocity', 'calvingThickness'])

        if self.damage is not None and self.damage != 'none':
            var_list.append('damage')

        if self.face_melt is True:
            var_list.append('faceMeltingThickness')

        if self.hydro is True:
            var_list.extend(['waterThickness', 'hydropotential', 'waterFlux',
                             'channelDischarge', 'channelArea'])

        if self.velo_solver in {'sia', 'none'}:
            compare_variables(test_case=self,
                              variables=var_list,
                              filename1=run_dir1 + '/output.nc',
                              filename2=run_dir2 + '/output.nc')

        elif self.velo_solver == 'FO':
            # validate thickness
            variable = ['thickness']
            l1_norm = 1.0e-10
            l2_norm = 1.0e-11
            linf_norm = 1.0e-11
            compare_variables(test_case=self, variables=variable,
                              filename1=run_dir1 + '/output.nc',
                              filename2=run_dir2 + '/output.nc',
                              l1_norm=l1_norm, l2_norm=l2_norm,
                              linf_norm=linf_norm, quiet=False)

            # validate speed
            variable = ['surfaceSpeed']
            l1_norm = 1.0e-15
            l2_norm = 1.0e-16
            linf_norm = 1.0e-17
            compare_variables(test_case=self, variables=variable,
                              filename1=run_dir1 + '/output.nc',
                              filename2=run_dir2 + '/output.nc',
                              l1_norm=l1_norm, l2_norm=l2_norm,
                              linf_norm=linf_norm, quiet=False)

            if 'calvingVelocity' in var_list:
                l1_norm = 1.0e-11
                l2_norm = 1.0e-11
                linf_norm = 1.0e-12
                compare_variables(test_case=self,
                                  variables=['calvingVelocity'],
                                  filename1=run_dir1 + '/output.nc',
                                  filename2=run_dir2 + '/output.nc',
                                  l1_norm=l1_norm, l2_norm=l2_norm,
                                  linf_norm=linf_norm, quiet=False)

            if 'calvingThickness' in var_list:
                l1_norm = 1.0e-10
                l2_norm = 1.0e-10
                linf_norm = 1.0e-11
                compare_variables(test_case=self,
                                  variables=['calvingThickness'],
                                  filename1=run_dir1 + '/output.nc',
                                  filename2=run_dir2 + '/output.nc',
                                  l1_norm=l1_norm, l2_norm=l2_norm,
                                  linf_norm=linf_norm, quiet=False)

            if 'damage' in var_list:
                l1_norm = 1.0e-11
                l2_norm = 1.0e-11
                linf_norm = 1.0e-12
                compare_variables(test_case=self,
                                  variables=['damage'],
                                  filename1=run_dir1 + '/output.nc',
                                  filename2=run_dir2 + '/output.nc',
                                  l1_norm=l1_norm, l2_norm=l2_norm,
                                  linf_norm=linf_norm, quiet=False)

            if 'faceMeltingThickness' in var_list:
                l1_norm = 1.0e-11
                l2_norm = 1.0e-11
                linf_norm = 1.0e-12
                compare_variables(test_case=self,
                                  variables=['faceMeltingThickness'],
                                  filename1=run_dir1 + '/output.nc',
                                  filename2=run_dir2 + '/output.nc',
                                  l1_norm=l1_norm, l2_norm=l2_norm,
                                  linf_norm=linf_norm, quiet=False)
