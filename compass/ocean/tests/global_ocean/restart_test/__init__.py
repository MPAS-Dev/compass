from compass.validate import compare_variables
from compass.ocean.tests.global_ocean.forward import ForwardTestCase, \
    ForwardStep


class RestartTest(ForwardTestCase):
    """
    A test case for performing two forward run, one without a restart and one
    with to make sure the results are identical
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
                         name='restart_test')
        module = __name__

        restart_time = {'split_explicit': '0001-01-01_04:00:00',
                        'RK4': '0001-01-01_00:10:00'}
        restart_filename = '../restarts/rst.{}.nc'.format(
            restart_time[time_integrator].replace(':', '.'))
        input_file = {'restart': restart_filename}
        output_file = {'full': restart_filename}
        for part in ['full', 'restart']:
            name = '{}_run'.format(part)
            step = ForwardStep(test_case=self, mesh=mesh, init=init,
                               time_integrator=time_integrator, name=name,
                               subdir=name, ntasks=4, openmp_threads=1)

            suffix = '{}.{}'.format(time_integrator.lower(), part)
            step.add_namelist_file(module, 'namelist.{}'.format(suffix))
            step.add_streams_file(module, 'streams.{}'.format(suffix))
            if part in input_file:
                step.add_input_file(filename=input_file[part])
            if part in output_file:
                step.add_output_file(filename=output_file[part])
            step.add_output_file(filename='output.nc')
            self.add_step(step)

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['temperature', 'salinity', 'layerThickness',
                     'normalVelocity']
        compare_variables(test_case=self, variables=variables,
                          filename1='full_run/output.nc',
                          filename2='restart_run/output.nc')
