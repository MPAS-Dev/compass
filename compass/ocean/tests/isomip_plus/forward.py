import os
import shutil
import xarray
import time

from compass.model import run_model
from compass.step import Step

from compass.ocean.tests.isomip_plus.evap import update_evaporation_flux
from compass.ocean.tests.isomip_plus.viz.plot import MoviePlotter


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
                 subdir=None, run_duration=None, time_varying_forcing=False):
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
                         ntasks=None, min_tasks=None, openmp_threads=1)

        # make sure output is double precision
        self.add_streams_file('compass.ocean.streams', 'streams.output')

        self.add_namelist_file('compass.ocean.tests.isomip_plus',
                               'namelist.forward_and_ssh_adjust')
        self.add_namelist_file('compass.ocean.tests.isomip_plus',
                               'namelist.forward')

        options = get_time_steps(resolution)

        if run_duration is not None:
            options['config_run_duration'] = run_duration

        self.add_namelist_options(options=options)

        self.add_streams_file('compass.ocean.streams',
                              'streams.land_ice_fluxes')

        template_replacements = {'output_interval': run_duration}
        self.add_streams_file('compass.ocean.tests.isomip_plus',
                              'streams.forward.template',
                              template_replacements=template_replacements)

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
                            target='../initial_state/culled_graph.info')
        self.add_input_file(
            filename='forcing_data_init.nc',
            target='../initial_state/init_mode_forcing_data.nc')

        self.add_input_file(
            filename='forcing_data.nc',
            target='forcing_data_init.nc')

        self.add_model_as_input()

        self.add_output_file('output.nc')
        self.add_output_file('land_ice_fluxes.nc')

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """
        run_model(self)

        if self.name == 'performance':
            # plot a few fields
            plot_folder = '{}/plots'.format(self.work_dir)
            if os.path.exists(plot_folder):
                shutil.rmtree(plot_folder)

            dsMesh = xarray.open_dataset(os.path.join(self.work_dir,
                                                      'init.nc'))
            ds = xarray.open_dataset(os.path.join(self.work_dir, 'output.nc'))

            section_y = self.config.getfloat('isomip_plus_viz', 'section_y')
            # show progress only if we're not writing to a log file
            show_progress = self.log_filename is None
            plotter = MoviePlotter(inFolder=self.work_dir,
                                   streamfunctionFolder=self.work_dir,
                                   outFolder=plot_folder, sectionY=section_y,
                                   dsMesh=dsMesh, ds=ds, expt=self.experiment,
                                   showProgress=show_progress)

            plotter.plot_3d_field_top_bot_section(
                ds.temperature, nameInTitle='temperature', prefix='temp',
                units='C', vmin=-2., vmax=1., cmap='cmo.thermal')

            plotter.plot_3d_field_top_bot_section(
                ds.salinity, nameInTitle='salinity', prefix='salin',
                units='PSU', vmin=33.8, vmax=34.7, cmap='cmo.haline')

        if self.name == 'simulation':
            update_evaporation_flux(in_forcing_file='forcing_data_init.nc',
                                    out_forcing_file='forcing_data_updated.nc',
                                    out_forcing_link='forcing_data.nc')

            replacements = {'config_do_restart': '.true.',
                            'config_start_time': "'file'"}
            self.update_namelist_at_runtime(replacements)


def get_time_steps(resolution):
    """
    Get the time step namelist replacements for the resolution

    Parameters
    ----------
    resolution : float
        The resolution in km

    Returns
    -------
    options : dict
        A dictionary with replacements for ``config_dt`` and ``config_brt_dt``
    """

    # 4 minutes at 2 km, and proportional to resolution
    dt = 2.*60*resolution

    # 10 sec at 2 km, and proportional to resolution
    btr_dt = 5.*resolution

    # https://stackoverflow.com/a/1384565/7728169
    # Note: this will drop any fractional seconds, which is usually okay
    dt = time.strftime('%H:%M:%S', time.gmtime(dt))

    btr_dt = time.strftime('%H:%M:%S', time.gmtime(btr_dt))

    return dict(config_dt="'{}'".format(dt),
                config_btr_dt="'{}'".format(btr_dt))
