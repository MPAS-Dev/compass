import xarray
import numpy
import os
import shutil

from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.translate import translate
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.cime.constants import constants

from compass.step import Step
from compass.ocean.vertical import init_vertical_coord
from compass.ocean.vertical.grid_1d import generate_1d_grid
from compass.ocean.iceshelf import compute_land_ice_pressure_and_draft
from compass.ocean.tests.isomip_plus.geom import process_input_geometry, \
    interpolate_geom, interpolate_ocean_mask
from compass.ocean.tests.isomip_plus.viz.plot import MoviePlotter


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for ISOMIP+ test cases

    Attributes
    ----------
    resolution : float
        The horizontal resolution (km) of the test case

    experiment : {'Ocean0', 'Ocean1', 'Ocean2'}
        The ISOMIP+ experiment

    vertical_coordinate : str
        The type of vertical coordinate (``z-star``, ``z-level``, etc.)

    time_varying_forcing : bool
        Whether the run includes time-varying land-ice forcing
    """
    def __init__(self, test_case, resolution, experiment, vertical_coordinate,
                 time_varying_forcing):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : float
            The horizontal resolution (km) of the test case

        experiment : {'Ocean0', 'Ocean1', 'Ocean2'}
            The ISOMIP+ experiment

        vertical_coordinate : str
            The type of vertical coordinate (``z-star``, ``z-level``, etc.)

        time_varying_forcing : bool
            Whether the run includes time-varying land-ice forcing
        """
        super().__init__(test_case=test_case, name='initial_state')
        self.resolution = resolution
        self.experiment = experiment
        self.vertical_coordinate = vertical_coordinate
        self.time_varying_forcing = time_varying_forcing

        if experiment in ['Ocean0', 'Ocean1']:
            self.add_input_file(filename='input_geometry.nc',
                                target='Ocean1_input_geom_v1.01.nc',
                                database='initial_condition_database')
        elif experiment == 'Ocean2':
            self.add_input_file(filename='input_geometry.nc',
                                target='Ocean2_input_geom_v1.01.nc',
                                database='initial_condition_database')
        else:
            raise ValueError('Unknown ISOMIP+ experiment {}'.format(
                experiment))

        for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info',
                     'initial_state.nc', 'init_mode_forcing_data.nc']:
            self.add_output_file(file)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger

        section = config['isomip_plus']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')
        filter_sigma = section.getfloat('topo_smoothing')*self.resolution
        min_ice_thickness = section.getfloat('min_ice_thickness')
        min_land_ice_fraction = section.getfloat('min_land_ice_fraction')
        draft_scaling = section.getfloat('draft_scaling')

        process_input_geometry('input_geometry.nc',
                               'input_geometry_processed.nc',
                               filterSigma=filter_sigma,
                               minIceThickness=min_ice_thickness,
                               scale=draft_scaling)

        dsMesh = make_planar_hex_mesh(nx=nx+2, ny=ny+2, dc=dc,
                                      nonperiodic_x=False, nonperiodic_y=False)
        translate(mesh=dsMesh, yOffset=-2*dc)
        write_netcdf(dsMesh, 'base_mesh.nc')

        dsGeom = xarray.open_dataset('input_geometry_processed.nc')

        min_ocean_fraction = config.getfloat('isomip_plus',
                                             'min_ocean_fraction')

        dsMask = interpolate_ocean_mask(dsMesh, dsGeom, min_ocean_fraction)
        dsMesh = cull(dsMesh, dsInverse=dsMask, logger=logger)
        dsMesh.attrs['is_periodic'] = 'NO'

        dsMesh = convert(dsMesh, graphInfoFileName='culled_graph.info',
                         logger=logger)
        write_netcdf(dsMesh, 'culled_mesh.nc')

        ds = interpolate_geom(dsMesh, dsGeom, min_ocean_fraction)

        for var in ['landIceFraction']:
            ds[var] = ds[var].expand_dims(dim='Time', axis=0)

        ds['landIceMask'] = \
            (ds.landIceFraction >= min_land_ice_fraction).astype(int)

        ref_density = constants['SHR_CONST_RHOSW']
        landIcePressure, landIceDraft = compute_land_ice_pressure_and_draft(
            ssh=ds.ssh, modify_mask=ds.ssh < 0., ref_density=ref_density)

        ds['landIcePressure'] = landIcePressure
        ds['landIceDraft'] = landIceDraft

        if self.time_varying_forcing:
            self._write_time_varying_forcing(ds_init=ds)

        ds['bottomDepth'] = -ds.bottomDepthObserved

        section = config['isomip_plus']

        min_column_thickness = section.getfloat('min_column_thickness')
        min_levels = section.getint('minimum_levels')

        interfaces = generate_1d_grid(config)

        # Deepen the bottom depth to maintain the minimum water-column
        # thickness
        min_depth = numpy.maximum(-ds.ssh + min_column_thickness,
                                  interfaces[min_levels+1])
        ds['bottomDepth'] = numpy.maximum(ds.bottomDepth, min_depth)

        init_vertical_coord(config, ds)

        ds['modifyLandIcePressureMask'] = \
            (ds['landIceFraction'] > 0.01).astype(int)

        max_bottom_depth = -config.getfloat('vertical_grid', 'bottom_depth')
        frac = (0. - ds.zMid) / (0. - max_bottom_depth)

        # compute T, S
        init_top_temp = section.getfloat('init_top_temp')
        init_bot_temp = section.getfloat('init_bot_temp')
        init_top_sal = section.getfloat('init_top_sal')
        init_bot_sal = section.getfloat('init_bot_sal')
        ds['temperature'] = (1.0 - frac) * init_top_temp + frac * init_bot_temp
        ds['salinity'] = (1.0 - frac) * init_top_sal + frac * init_bot_sal

        # compute coriolis
        coriolis_parameter = section.getfloat('coriolis_parameter')

        ds['fCell'] = coriolis_parameter*xarray.ones_like(ds.xCell)
        ds['fEdge'] = coriolis_parameter*xarray.ones_like(ds.xEdge)
        ds['fVertex'] = coriolis_parameter*xarray.ones_like(ds.xVertex)

        normalVelocity = xarray.zeros_like(ds.xEdge)
        normalVelocity = normalVelocity.broadcast_like(ds.refBottomDepth)
        normalVelocity = normalVelocity.transpose('nEdges', 'nVertLevels')
        ds['normalVelocity'] = normalVelocity.expand_dims(dim='Time', axis=0)

        write_netcdf(ds, 'initial_state.nc')

        plot_folder = '{}/plots'.format(self.work_dir)
        if os.path.exists(plot_folder):
            shutil.rmtree(plot_folder)

        # plot a few fields
        section_y = config.getfloat('isomip_plus_viz', 'section_y')

        # show progress only if we're not writing to a log file
        show_progress = self.log_filename is None

        plotter = MoviePlotter(inFolder=self.work_dir,
                               streamfunctionFolder=self.work_dir,
                               outFolder=plot_folder, expt=self.experiment,
                               sectionY=section_y, dsMesh=ds, ds=ds,
                               showProgress=show_progress)

        plotter.plot_3d_field_top_bot_section(
            ds.zMid, nameInTitle='zMid', prefix='zmid', units='m',
            vmin=-720., vmax=0., cmap='cmo.deep_r')

        plotter.plot_3d_field_top_bot_section(
            ds.temperature, nameInTitle='temperature', prefix='temp',
            units='C', vmin=-2., vmax=1., cmap='cmo.thermal')

        plotter.plot_3d_field_top_bot_section(
            ds.salinity, nameInTitle='salinity', prefix='salin',
            units='PSU', vmin=33.8, vmax=34.7, cmap='cmo.haline')

        # compute restoring
        dsForcing = xarray.Dataset()

        restore_top_temp = section.getfloat('restore_top_temp')
        restore_bot_temp = section.getfloat('restore_bot_temp')
        restore_top_sal = section.getfloat('restore_top_sal')
        restore_bot_sal = section.getfloat('restore_bot_sal')
        dsForcing['temperatureInteriorRestoringValue'] = \
            (1.0 - frac) * restore_top_temp + frac * restore_bot_temp
        dsForcing['salinityInteriorRestoringValue'] = \
            (1.0 - frac) * restore_top_sal + frac * restore_bot_sal

        restore_rate = section.getfloat('restore_rate')
        restore_xmin = section.getfloat('restore_xmin')
        restore_xmax = section.getfloat('restore_xmax')
        frac = numpy.maximum(
            (ds.xCell - restore_xmin)/(restore_xmax-restore_xmin), 0.)
        frac = frac.broadcast_like(dsForcing.temperatureInteriorRestoringValue)

        # convert from 1/days to 1/s
        dsForcing['temperatureInteriorRestoringRate'] = \
            frac * restore_rate / constants['SHR_CONST_CDAY']
        dsForcing['salinityInteriorRestoringRate'] = \
            dsForcing.temperatureInteriorRestoringRate

        # compute "evaporation"
        restore_evap_rate = section.getfloat('restore_evap_rate')

        mask = numpy.logical_and(ds.xCell >= restore_xmin,
                                 ds.xCell <= restore_xmax)
        mask = mask.expand_dims(dim='Time', axis=0)
        # convert to m/s, negative for evaporation rather than precipitation
        evap_rate = -restore_evap_rate/(constants['SHR_CONST_CDAY']*365)
        # PSU*m/s to kg/m^2/s
        sflux_factor = 1.
        # C*m/s to W/m^2
        hflux_factor = 1./(ref_density*constants['SHR_CONST_CPSW'])
        dsForcing['evaporationFlux'] = mask*ref_density*evap_rate
        dsForcing['seaIceSalinityFlux'] = \
            mask*evap_rate*restore_top_sal/sflux_factor
        dsForcing['seaIceHeatFlux'] = \
            mask*evap_rate*restore_top_temp/hflux_factor

        write_netcdf(dsForcing, 'init_mode_forcing_data.nc')

    def _write_time_varying_forcing(self, ds_init):
        """
        Write time-varying land-ice forcing and update the initial condition
        """

        config = self.config
        dates = config.get('isomip_plus_forcing', 'dates')
        dates = [date.ljust(64) for date in dates.replace(',', ' ').split()]
        scales = config.get('isomip_plus_forcing', 'scales')
        scales = [float(scale) for scale in scales.replace(',', ' ').split()]

        ds_out = xarray.Dataset()
        ds_out['xtime'] = ('Time', dates)
        ds_out['xtime'] = ds_out.xtime.astype('S')

        landIceDraft = list()
        landIcePressure = list()
        landIceFraction = list()

        for scale in scales:
            landIceDraft.append(scale*ds_init.landIceDraft)
            landIcePressure.append(scale*ds_init.landIcePressure)
            landIceFraction.append(ds_init.landIceFraction)

        ds_out['landIceDraftForcing'] = xarray.concat(landIceDraft, 'Time')
        ds_out.landIceDraftForcing.attrs['units'] = 'm'
        ds_out.landIceDraftForcing.attrs['long_name'] = \
            'The approximate elevation of the land ice-ocean interface'
        ds_out['landIcePressureForcing'] = \
            xarray.concat(landIcePressure, 'Time')
        ds_out.landIcePressureForcing.attrs['units'] = 'm'
        ds_out.landIcePressureForcing.attrs['long_name'] = \
            'Pressure from the weight of land ice at the ice-ocean interface'
        ds_out['landIceFractionForcing'] = \
            xarray.concat(landIceFraction, 'Time')
        ds_out.landIceFractionForcing.attrs['units'] = 'unitless'
        ds_out.landIceFractionForcing.attrs['long_name'] = \
            'The fraction of each cell covered by land ice'
        write_netcdf(ds_out, 'land_ice_forcing.nc')

        ds_init['landIceDraft'] = scales[0]*ds_init.landIceDraft
        ds_init['ssh'] = ds_init.landIceDraft
        ds_init['landIcePressure'] = scales[0]*ds_init.landIcePressure
