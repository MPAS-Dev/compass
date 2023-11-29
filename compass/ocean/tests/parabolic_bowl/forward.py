import time

from compass.model import run_model
from compass.step import Step


class Forward(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of parabolic bowl
    test cases.
    """
    def __init__(self, test_case, resolution,
                 name, use_lts,
                 ramp_type='ramp', coord_type='single_layer',
                 wetdry='standard'):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : float
            The resolution of the test case

        name : str
            The name of the test case

        use_lts : bool
            Whether local time-stepping is used

        subdir : str, optional
            The subdirectory for the step.  The default is ``name``

        coord_type : str, optional
            Vertical coordinate configuration

        ramp_type : str, optional
            Vertical coordinate configuration
        """

        self.resolution = resolution

        super().__init__(test_case=test_case, name=name)

        self.add_namelist_file('compass.ocean.tests.parabolic_bowl',
                               'namelist.forward')

        res_name = f'{resolution}km'
        self.add_namelist_file('compass.ocean.tests.parabolic_bowl',
                               f'namelist.{coord_type}.forward')

        if ramp_type == 'ramp':
            self.add_namelist_file('compass.ocean.tests.parabolic_bowl',
                                   'namelist.ramp.forward')

        if use_lts:
            self.add_namelist_options(
                {'config_time_integrator': "'LTS'"})
            self.add_namelist_options(
                {'config_dt_scaling_LTS': "4"})
            self.add_namelist_options(
                {'config_number_of_time_levels': "4"})
            self.add_streams_file('compass.ocean.tests.parabolic_bowl.lts',
                                  'streams.forward')
            input_path = f'../lts_regions_{res_name}'
            self.add_input_file(filename='mesh.nc',
                                target=f'{input_path}/lts_mesh.nc')
            self.add_input_file(filename='graph.info',
                                target=f'{input_path}/lts_graph.info')
            self.add_input_file(filename='init.nc',
                                target=f'{input_path}/lts_ocean.nc')

        else:
            self.add_streams_file('compass.ocean.tests.parabolic_bowl',
                                  'streams.forward')
            input_path = f'../initial_state_{res_name}'
            self.add_input_file(filename='mesh.nc',
                                target=f'{input_path}/culled_mesh.nc')
            self.add_input_file(filename='init.nc',
                                target=f'{input_path}/ocean.nc')
            self.add_input_file(filename='graph.info',
                                target=f'{input_path}/culled_graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='output.nc')

    def setup(self):
        """
        Set namelist options based on config options
        """
        dt = self.get_dt()
        self.add_namelist_options({'config_dt': dt})
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
        # update dt in case the user has changed dt_per_km
        dt = self.get_dt()
        self.update_namelist_at_runtime(options={'config_dt': dt},
                                        out_name='namelist.ocean')

        run_model(self)

    def get_dt(self):
        """
        Get the time step

        Returns
        -------
        dt : str
            the time step in HH:MM:SS
        """
        config = self.config
        # dt is proportional to resolution
        dt_per_km = config.getfloat('parabolic_bowl', 'dt_per_km')

        dt = dt_per_km * self.resolution
        # https://stackoverflow.com/a/1384565/7728169
        dt = time.strftime('%H:%M:%S', time.gmtime(dt))

        return dt

    def _get_resources(self):
        """ get the these properties from the config options """
        config = self.config
        self.ntasks = config.getint('parabolic_bowl',
                                    f'{self.resolution}km_ntasks')
        self.min_tasks = config.getint('parabolic_bowl',
                                       f'{self.resolution}km_min_tasks')
        self.openmp_threads = 1
