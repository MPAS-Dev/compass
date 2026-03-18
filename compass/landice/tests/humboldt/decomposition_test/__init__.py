from compass.landice.tests.humboldt.run_model import RunModel
from compass.parallel import get_available_parallel_resources
from compass.testcase import TestCase
from compass.validate import compare_variables


class DecompositionTest(TestCase):
    """
    A test case for performing two MALI runs of a Humboldt setup with
    different decompositions. The larger decomposition targets 32 tasks,
    subject to available resources, and the smaller decomposition is roughly
    half of the larger one. The test case verifies that results are identical
    or close to identical.

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

    proc_list : list of int
        The pair of processor counts used in the decomposition comparison

    run_dirs : list of str
        The names of the subdirectories for the two decomposition runs
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
        self.depth_integrated = depth_integrated
        self.proc_list = None
        self.run_dirs = None
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

    def configure(self):
        """
        Choose decomposition sizes from framework-detected resources and add
        run steps.

        The larger decomposition targets up to 32 tasks. FO runs require at
        least 10 tasks; all others require at least 2 tasks.
        """
        available_resources = get_available_parallel_resources(self.config)
        # Target a max of 32 tasks, but use fewer if not available.
        target_max_tasks = 32
        # FO solver required more resources to be time-effective to run
        if self.velo_solver == 'FO':
            smallest_acceptable_max_tasks = 10
        else:
            # Need at least 2 tasks to test decomposition.
            smallest_acceptable_max_tasks = 2
        max_tasks = max(
            smallest_acceptable_max_tasks,
            min(target_max_tasks, available_resources['cores']))
        # Note: Failing when this many tasks are unavailable is
        # desired behavior for decomposition testing.

        low_tasks = max(1, max_tasks // 2)
        self.proc_list = [low_tasks, max_tasks]

        self.run_dirs = []
        for procs in self.proc_list:
            name = '{}proc_run'.format(procs)
            if name in self.run_dirs:
                name = '{}_{}'.format(name, len(self.run_dirs) + 1)
            self.run_dirs.append(name)
            self.add_step(
                RunModel(test_case=self, name=name, subdir=name, ntasks=procs,
                         min_tasks=procs,
                         openmp_threads=1, velo_solver=self.velo_solver,
                         calving_law=self.calving_law,
                         damage=self.damage,
                         face_melt=self.face_melt,
                         depth_integrated=self.depth_integrated,
                         hydro=self.hydro,
                         mesh_type=self.mesh_type))

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        run_dir1 = self.run_dirs[0]
        run_dir2 = self.run_dirs[1]

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
