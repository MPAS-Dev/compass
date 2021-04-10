from compass.validate import compare_variables
from compass.ocean.tests.global_ocean.forward import ForwardTestCase


class DynamicAdjustment(ForwardTestCase):
    """
    A parent test case for performing dynamic adjustment (dissipating
    fast-moving waves) from an MPAS-Ocean initial condition.

    The final stage of the dynamic adjustment is assumed to be called
    ``simulation``, and is expected to have a file ``output.nc`` that can be
    compared against a baseline.

    Attributes
    ----------
    restart_filenames : list of str
        A list of restart files from each dynamic-adjustment step
    """

    def __init__(self, test_group, mesh, init, time_integrator,
                 restart_filenames):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.global_ocean.GlobalOcean
            The global ocean test group that this test case belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that produces the mesh for this run

        init : compass.ocean.tests.global_ocean.init.Init
            The test case that produces the initial condition for this run

        time_integrator : {'split_explicit', 'RK4'}
            The time integrator to use for the forward run

        restart_filenames : list of str
            A list of restart files from each dynamic-adjustment step
        """
        super().__init__(test_group=test_group, mesh=mesh, init=init,
                         time_integrator=time_integrator,
                         name='dynamic_adjustment')

        self.restart_filenames = restart_filenames

    def run(self):
        """
        Run each step of the testcase
        """
        # get cores, threads from config options and run the steps
        super().run()

        variables = ['temperature', 'salinity', 'layerThickness',
                     'normalVelocity']

        compare_variables(variables, self.config, work_dir=self.work_dir,
                          filename1='simulation/output.nc')
