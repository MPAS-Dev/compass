from compass.model import run_model
from compass.ocean.tests.isomip_plus.evap import update_evaporation_flux
from compass.ocean.time import get_time_interval_string
from compass.step import Step


class Forward(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of ice-shelf 2D test
    cases.

    Attributes
    ----------
    resolution : float
        The horizontal resolution (km) of the test case

    experiment : {'Ocean0', 'Ocean1', 'Ocean2'}
        The ISOMIP+ experiment

    """
    def __init__(self, test_case, resolution, experiment, name='forward',
                 subdir=None, run_duration=None, vertical_coordinate='z-star',
                 tidal_forcing=False, time_varying_forcing=False,
                 thin_film_present=False):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : float
            The horizontal resolution (km) of the test case

        experiment : {'Ocean0', 'Ocean1', 'Ocean2'}
            The ISOMIP+ experiment

        name : str, optional
            the name of the test case

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        run_duration : str, optional
            The duration of the run

        time_varying_forcing : bool, optional
            Whether the run includes time-varying land-ice forcing
        """
        self.resolution = resolution
        self.experiment = experiment
        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         ntasks=None, min_tasks=None, openmp_threads=None)

        # make sure output is double precision
        self.add_streams_file('compass.ocean.streams', 'streams.output')

        self.add_namelist_file('compass.ocean.tests.isomip_plus',
                               'namelist.forward_and_ssh_adjust')
        self.add_namelist_file('compass.ocean.tests.isomip_plus',
                               'namelist.forward')

        options = dict()
        if run_duration is not None:
            options['config_run_duration'] = run_duration

        self.add_namelist_options(options=options)

        self.add_streams_file('compass.ocean.streams',
                              'streams.land_ice_fluxes')

        if tidal_forcing:
            output_interval = "0000-00-00_02:00:00"
        else:
            output_interval = run_duration
        template_replacements = {'output_interval': output_interval}

        self.add_streams_file('compass.ocean.tests.isomip_plus',
                              'streams.forward.template',
                              template_replacements=template_replacements)

        if vertical_coordinate == 'single_layer':
            self.add_namelist_file(
                'compass.ocean.tests.isomip_plus',
                'namelist.single_layer.forward_and_ssh_adjust')
        if tidal_forcing:
            self.add_namelist_file('compass.ocean.tests.isomip_plus',
                                   'namelist.tidal_forcing.forward')

        if thin_film_present:
            self.add_namelist_file('compass.ocean.tests.isomip_plus',
                                   'namelist.thin_film.forward_and_ssh_adjust')
        if time_varying_forcing:
            self.add_namelist_file('compass.ocean.tests.isomip_plus',
                                   'namelist.time_varying_forcing')
            self.add_streams_file('compass.ocean.tests.isomip_plus',
                                  'streams.time_varying_forcing')
            self.add_input_file(
                filename='land_ice_forcing.nc',
                target='../initial_state/land_ice_forcing.nc')

        self.add_input_file(filename='init.nc',
                            target='../ssh_adjustment/adjusted_init.nc')
        self.add_input_file(filename='graph.info',
                            target='../cull_mesh/culled_graph.info')
        self.add_input_file(
            filename='forcing_data_init.nc',
            target='../initial_state/init_mode_forcing_data.nc')

        self.add_input_file(
            filename='forcing_data.nc',
            target='forcing_data_init.nc')

        self.add_model_as_input()

        self.add_output_file('output.nc')
        self.add_output_file('land_ice_fluxes.nc')

    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies
        """
        self._get_resources()

    def constrain_resources(self, available_resources):
        """
        Update resources at runtime from config options
        """
        self._get_resources()
        super().constrain_resources(available_resources)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        resolution = self.resolution

        dt_per_km = config.getfloat('isomip_plus', 'dt_per_km')
        dt_btr_per_km = config.getfloat('isomip_plus', 'dt_btr_per_km')

        dt = get_time_interval_string(seconds=dt_per_km * resolution)
        btr_dt = get_time_interval_string(seconds=dt_btr_per_km * resolution)

        options = dict(config_dt=f"'{dt}'",
                       config_btr_dt=f"'{btr_dt}'")
        self.update_namelist_at_runtime(options)

        run_model(self)

        if self.name == 'simulation':
            update_evaporation_flux(
                in_forcing_file='forcing_data_init.nc',
                out_forcing_file='forcing_data_updated.nc',
                out_forcing_link='forcing_data.nc')

            replacements = {'config_do_restart': '.true.',
                            'config_start_time': "'file'"}
            self.update_namelist_at_runtime(replacements)

    def _get_resources(self):
        """
        Get resources (ntasks, min_tasks, and openmp_threads) from the config
        options
        """
        config = self.config
        self.ntasks = config.getint('isomip_plus', 'forward_ntasks')
        self.min_tasks = config.getint('isomip_plus', 'forward_min_tasks')
        self.openmp_threads = config.getint('isomip_plus', 'forward_threads')
