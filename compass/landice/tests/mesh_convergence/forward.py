import math
from importlib.resources import contents

from compass.model import run_model
from compass.step import Step


class Forward(Step):
    """
    A step for performing forward MALI runs as part of a mesh
    convergence test case

    Attributes
    ----------
    resolution : int
        The resolution of the (uniform) mesh in km
    """

    def __init__(self, test_case, resolution):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.landice.tests.mesh_convergence.convergence_test_case.ConvergenceTestCase
            The test case this step belongs to

        resolution : int
            The resolution of the (uniform) mesh in km
        """  # noqa: E501
        super().__init__(test_case=test_case,
                         name='{}km_forward'.format(resolution),
                         subdir='{}km/forward'.format(resolution))

        self.resolution = resolution

        self.add_namelist_file(
            'compass.landice.tests.mesh_convergence', 'namelist.forward')

        self.add_streams_file('compass.landice.tests.mesh_convergence',
                              'streams.forward')

        module = self.test_case.__module__
        mesh_package_contents = list(contents(module))
        if 'namelist.forward' in mesh_package_contents:
            self.add_namelist_file(module, 'namelist.forward')
        if 'streams.forward' in mesh_package_contents:
            self.add_streams_file(module, 'streams.forward')

        self.add_input_file(filename='init.nc',
                            target='../init/initial_state.nc')
        self.add_input_file(filename='graph.info',
                            target='../init/graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='output.nc')

    def setup(self):
        """
        Set namelist options based on config options
        """

        namelist_options, stream_replacements = self.get_dt_duration()
        self.add_namelist_options(namelist_options)

        self.add_streams_file('compass.landice.tests.mesh_convergence',
                              'streams.template',
                              template_replacements=stream_replacements)
        self._get_resources()

    def constrain_resources(self, available_resources):
        """
        Update resources at runtime from config options
        """
        self._get_resources()
        super().constrain_resources(available_resources)

    def run(self):
        """
        Run this step of the testcase
        """
        namelist_options, stream_replacements = self.get_dt_duration()
        self.update_namelist_at_runtime(
            options=namelist_options,
            out_name='namelist.landice')

        self.update_streams_at_runtime(
            'compass.landice.tests.mesh_convergence',
            'streams.template', template_replacements=stream_replacements,
            out_name='streams.landice')

        run_model(self)

    def get_dt_duration(self):
        """
        Get the time step and run duration as namelist options from config
        options

        Returns
        -------
        options : dict
            Namelist options to update
        """

        sec_in_yr = 3600.0 * 24.0 * 365.0

        config = self.config
        duration = config.getint('mesh_convergence', 'duration')
        dur_sec = duration * sec_in_yr
        target_velocity = config.getfloat('mesh_convergence',
                                          'target_velocity')
        resolutions = config.getlist('mesh_convergence', 'resolutions',
                                     dtype=int)
        max_res = max(resolutions)
        # calculate dt in s for the resolution in km and velo in m/yr
        # apply ceil to ensure dt * nt actually exceeds duration
        dt_max_res_cfl = math.ceil(max_res * 1000.0 / target_velocity *
                                   sec_in_yr)
        nt_max_res = math.ceil(dur_sec / dt_max_res_cfl)
        dt_max_res = math.ceil(dur_sec / nt_max_res)
        print(f'max res: nt={nt_max_res}, dt={dt_max_res}, '
              f'err={abs(dur_sec - nt_max_res * dt_max_res) / dur_sec}')

        dt = dt_max_res * self.resolution / max_res
        nt = math.ceil(dur_sec / dt)
        print(f'{self.resolution}km res: nt={nt}, dt={dt}, '
              f'err={abs(dur_sec - nt * dt) / dur_sec}')

        # check that dt*nt is acceptably close to duration
        time_err = abs(dur_sec - nt * dt) / dur_sec
        assert time_err < 1e-4, f'nt*dt differs from duration by {time_err}'

        # the duration (years) of the run
        duration = f'{duration:05d}-00-00_00:00:00'

        namelist_replacements = {'config_dt': f"'{dt}'",
                                 'config_run_duration': f"'{duration}'"}

        stream_replacements = {'output_interval': duration}

        return namelist_replacements, stream_replacements

    def _get_resources(self):
        config = self.config
        resolution = self.resolution
        self.ntasks = config.getint('mesh_convergence',
                                    f'{resolution}km_ntasks')
        self.min_tasks = config.getint('mesh_convergence',
                                       f'{resolution}km_min_tasks')
