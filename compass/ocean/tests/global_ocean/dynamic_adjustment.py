import os
import xarray
import glob

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

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        config = self.config
        variables = ['temperature', 'salinity', 'layerThickness',
                     'normalVelocity']

        compare_variables(test_case=self, variables=variables,
                          filename1='simulation/output.nc')

        temp_max = config.getfloat('dynamic_adjustment', 'temperature_max')
        max_values = {'temperatureMax': temp_max}

        for step_name in self.steps_to_run:
            step = self.steps[step_name]
            step_path = os.path.join(self.work_dir,  step.subdir)
            global_stats_path = os.path.join(step_path, 'analysis_members',
                                             'globalStats.*.nc')
            global_stats_path = glob.glob(global_stats_path)
            for filename in global_stats_path:
                ds = xarray.open_dataset(filename)
                for var, max_value in max_values.items():
                    max_in_global_stats = ds[var].max().values
                    if max_in_global_stats > max_value:
                        raise ValueError(
                            f'Max of {var} > allowed threshold: '
                            f'{max_in_global_stats} > {max_value} '
                            f'in {filename}')
