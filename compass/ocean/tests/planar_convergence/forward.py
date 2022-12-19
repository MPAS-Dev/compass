import time
from datetime import datetime, timedelta
from importlib.resources import contents

from compass.model import run_model
from compass.step import Step


class Forward(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of a planar
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
        test_case : compass.ocean.tests.planar_convergence.convergence_test_case.ConvergenceTestCase
            The test case this step belongs to

        resolution : int
            The resolution of the (uniform) mesh in km
        """
        super().__init__(test_case=test_case,
                         name='{}km_forward'.format(resolution),
                         subdir='{}km/forward'.format(resolution))

        self.resolution = resolution

        self.add_namelist_file(
            'compass.ocean.tests.planar_convergence', 'namelist.forward')

        self.add_streams_file('compass.ocean.tests.planar_convergence',
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
        Set namelist options base on config options
        """

        namelist_options, stream_replacements = self.get_dt_duration()
        self.add_namelist_options(namelist_options)

        self.add_streams_file('compass.ocean.tests.planar_convergence',
                              'streams.template',
                              template_replacements=stream_replacements)
        self._get_resources()

    def constrain_resources(self, available_cores):
        """
        Update resources at runtime from config options
        """
        self._get_resources()
        super().constrain_resources(available_cores)

    def run(self):
        """
        Run this step of the testcase
        """
        namelist_options, stream_replacements = self.get_dt_duration()
        self.update_namelist_at_runtime(
            options=namelist_options,
            out_name='namelist.ocean')

        self.update_streams_at_runtime(
            'compass.ocean.tests.planar_convergence',
            'streams.template', template_replacements=stream_replacements,
            out_name='streams.ocean')

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
        config = self.config
        # dt is proportional to resolution: default 30 seconds per km
        dt_1km = config.getint('planar_convergence', 'dt_1km')

        dt = dt_1km * self.resolution
        # https://stackoverflow.com/a/1384565/7728169
        dt = time.strftime('%H:%M:%S', time.gmtime(dt))

        # the duration (hours) of the run
        duration = \
            int(3600 * config.getfloat('planar_convergence', 'duration'))
        delta = timedelta(seconds=duration)
        hours = delta.seconds//3600
        minutes = delta.seconds//60 % 60
        seconds = delta.seconds % 60
        duration = f'{delta.days:03d}_{hours:02d}:{minutes:02d}:{seconds:02d}'

        namelist_replacements = {'config_dt': f"'{dt}'",
                                 'config_run_duration': f"'{duration}'"}

        stream_replacements = {'output_interval': duration}

        return namelist_replacements, stream_replacements

    def _get_resources(self):
        config = self.config
        resolution = self.resolution
        self.ntasks = config.getint(f'planar_convergence',
                                    f'{resolution}km_ntasks')
        self.min_tasks = config.getint(f'planar_convergence',
                                       f'{resolution}km_min_tasks')
