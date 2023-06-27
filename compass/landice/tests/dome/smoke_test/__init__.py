from compass.landice.tests.dome.run_model import RunModel
from compass.landice.tests.dome.setup_mesh import SetupMesh
from compass.landice.tests.dome.visualize import Visualize
from compass.testcase import TestCase


class SmokeTest(TestCase):
    """
    The default test case for the dome test group simply creates the mesh and
    initial condition, then performs a short forward run on 4 cores.

    Attributes
    ----------
    mesh_type : str
        The resolution or type of mesh of the test case

    velo_solver : {'sia', 'FO'}
        The velocity solver to use for the test case

    advection_type : {'fo', 'fct'}
        The type of advection to use for thickness and tracers
    """

    def __init__(self, test_group, velo_solver, mesh_type, advection_type):
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

        advection_type : {'fo', 'fct'}
            The type of advection to use for thickness and tracers
        """
        name = 'smoke_test'
        self.mesh_type = mesh_type
        self.velo_solver = velo_solver
        self.advection_type = advection_type
        subdir = '{}/{}_{}_{}'.format(mesh_type, velo_solver.lower(),
                                      advection_type, name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(
            SetupMesh(test_case=self, mesh_type=mesh_type))

        step = RunModel(test_case=self, ntasks=4, openmp_threads=1,
                        name='run_step', velo_solver=velo_solver,
                        mesh_type=mesh_type)
        if velo_solver == 'sia':
            step.add_namelist_options(
                {'config_run_duration': "'0200-00-00_00:00:00'"})
        if advection_type == 'fct':
            step.add_namelist_options(
                {'config_thickness_advection': "'fct'",
                 'config_tracer_advection': "'fct'"})
        self.add_step(step)

        step = Visualize(test_case=self, mesh_type=mesh_type)
        self.add_step(step, run_by_default=False)

    # no configure() method is needed because we will use the default dome
    # config options

    # no run() method is needed because we're doing the default: running all
    # steps
