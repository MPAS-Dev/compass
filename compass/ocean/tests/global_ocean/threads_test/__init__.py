from compass.validate import compare_variables
from compass.ocean.tests.global_ocean.forward import ForwardTestCase, \
    ForwardStep


class ThreadsTest(ForwardTestCase):
    """
    A test case for performing two short forward runs to make sure the results
    are identical with 1 and 2 thread per MPI process
    """

    def __init__(self, test_group, mesh, init, time_integrator):
        """
        Create test case

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
        """
        super().__init__(test_group=test_group, mesh=mesh, init=init,
                         time_integrator=time_integrator,
                         name='threads_test')
        for threads in [1, 2]:
            name = '{}thread'.format(threads)
            self.add_step(
                ForwardStep(test_case=self, mesh=mesh, init=init,
                            time_integrator=time_integrator, name=name,
                            subdir=name, cores=4, threads=threads))

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['temperature', 'salinity', 'layerThickness',
                     'normalVelocity']
        compare_variables(test_case=self, variables=variables,
                          filename1='1thread/output.nc',
                          filename2='2thread/output.nc')
