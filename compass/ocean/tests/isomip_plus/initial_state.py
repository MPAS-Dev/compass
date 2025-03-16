import os
import shutil

import cmocean  # noqa: F401
import numpy as np
import xarray as xr
from mpas_tools.cime.constants import constants
from mpas_tools.io import write_netcdf

from compass.ocean.iceshelf import (
    compute_land_ice_draft_from_pressure,
    compute_land_ice_pressure_from_draft,
    compute_land_ice_pressure_from_thickness,
)
from compass.ocean.tests.isomip_plus.geom import interpolate_geom
from compass.ocean.tests.isomip_plus.viz.plot import MoviePlotter
from compass.ocean.vertical import init_vertical_coord
from compass.step import Step


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

    thin_film_present: bool
        Whether the run includes a thin film below grounded ice
    """
    def __init__(self, test_case, resolution, experiment, vertical_coordinate,
                 time_varying_forcing, thin_film_present, include_viz=False):
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

        thin_film_present: bool
            Whether the run includes a thin film below grounded ice
        """
        super().__init__(test_case=test_case, name='initial_state')
        self.include_viz = include_viz
        self.resolution = resolution
        self.experiment = experiment
        self.vertical_coordinate = vertical_coordinate
        self.time_varying_forcing = time_varying_forcing
        self.thin_film_present = thin_film_present

        self.add_input_file(
            filename='input_geometry_processed.nc',
            target='../process_geom/input_geometry_processed.nc')

        self.add_input_file(
            filename='culled_mesh.nc',
            target='../cull_mesh/culled_mesh.nc')

        for file in ['initial_state.nc', 'init_mode_forcing_data.nc']:
            self.add_output_file(file)

    def run(self):
        """
        Run this step of the test case
        """
        ds, frac = self._compute_initial_condition()
        self._compute_restoring(ds, frac)
        if self.include_viz:
            self._plot(ds)

    def _compute_initial_condition(self):
        config = self.config
        thin_film_present = self.thin_film_present

        if self.vertical_coordinate == 'single_layer':
            config.set('vertical_grid', 'vert_levels', '1',
                       comment='Number of vertical levels')
            config.set('vertical_grid', 'coord_type', 'z-level')
        section = config['isomip_plus']
        min_land_ice_fraction = section.getfloat('min_land_ice_fraction')
        min_ocean_fraction = section.getfloat('min_ocean_fraction')
        ice_density = section.getfloat('ice_density')

        ds_geom = xr.open_dataset('input_geometry_processed.nc')
        ds_mesh = xr.open_dataset('culled_mesh.nc')

        ds = interpolate_geom(ds_mesh, ds_geom, min_ocean_fraction,
                              thin_film_present)

        ds['bottomDepth'] = -ds.bottomDepthObserved

        ds['landIceFraction'] = \
            ds['landIceFraction'].expand_dims(dim='Time', axis=0)
        ds['landIceFloatingFraction'] = \
            ds['landIceFloatingFraction'].expand_dims(dim='Time', axis=0)

        ds['sshAdjustmentMask'] = \
            (ds['landIceFraction'] > 0.01).astype(int)

        # This inequality needs to be > rather than >= to ensure correctness
        # when min_land_ice_fraction = 0
        mask = ds.landIceFraction > min_land_ice_fraction

        floating_mask = np.logical_and(
            ds.landIceFloatingFraction > 0,
            ds.landIceFraction > min_land_ice_fraction)

        ds['landIceMask'] = mask.astype(int)
        ds['landIceFloatingMask'] = floating_mask.astype(int)

        ds['landIceFraction'] = xr.where(mask, ds.landIceFraction, 0.)

        if thin_film_present:  # compute full pressure in grounded regions
            land_ice_thickness = ds.landIceThickness
            land_ice_pressure_unscaled = \
                compute_land_ice_pressure_from_thickness(
                    land_ice_thickness=land_ice_thickness,
                    modify_mask=ds.ssh < 0.,
                    land_ice_density=ice_density)
        else:  # assume floatation pressure everywhere
            land_ice_draft = ds.ssh
            land_ice_pressure_unscaled = compute_land_ice_pressure_from_draft(
                land_ice_draft=land_ice_draft, modify_mask=land_ice_draft < 0.)

        if self.time_varying_forcing:
            scales = config.get('isomip_plus_forcing', 'scales')
            scales = [float(scale)
                      for scale in scales.replace(',', ' ').split()]
            land_ice_pressure = land_ice_pressure_unscaled * scales[0]
        else:
            land_ice_pressure = land_ice_pressure_unscaled

        # Previously, we didn't apply max draft at seafloor but we want this
        # for W&D
        # This assumes that for MALI forcing, landIceDraft = bottomDepth in
        # grounded regions
        if thin_film_present:
            modify_mask = ds.bottomDepth > 0.
            land_ice_draft = compute_land_ice_draft_from_pressure(
                land_ice_pressure=land_ice_pressure,
                modify_mask=modify_mask)
            land_ice_draft = np.maximum(land_ice_draft,
                                        -ds.bottomDepth)
            ds['ssh'] = land_ice_draft

        ds['landIcePressure'] = land_ice_pressure.expand_dims(
            dim='Time', axis=0)

        if self.time_varying_forcing:
            self._write_time_varying_forcing(
                ds_init=ds,
                ice_density=ice_density,
                land_ice_pressure_unscaled=land_ice_pressure_unscaled)

        # Deepen the bottom depth to maintain the minimum water-column
        # thickness
        section = config['isomip_plus']
        min_column_thickness = section.getfloat('min_column_thickness')
        min_layer_thickness = section.getfloat('min_layer_thickness')
        min_levels = section.getint('minimum_levels')
        min_column_thickness = max(min_column_thickness,
                                   min_levels * min_layer_thickness)
        min_depth = -ds.ssh + min_column_thickness
        ds['bottomDepth'] = np.maximum(ds.bottomDepth, min_depth)
        print(f'Adjusted bottomDepth for '
              f'{np.sum(ds.bottomDepth.values < min_depth.values)} cells '
              f'to achieve minimum column thickness of {min_column_thickness}')

        # Initialize vertical coordinate without time dimension
        ds = ds.isel(Time=0)
        init_vertical_coord(config, ds)

        # compute T, S
        init_top_temp = section.getfloat('init_top_temp')
        init_bot_temp = section.getfloat('init_bot_temp')
        init_top_sal = section.getfloat('init_top_sal')
        init_bot_sal = section.getfloat('init_bot_sal')

        if self.vertical_coordinate == 'single_layer':
            # Initialize constant T,S
            ds['temperature'] = init_bot_temp * xr.ones_like(ds.zmid)
            ds['salinity'] = init_bot_sal * xr.ones_like(ds.zmid)
        else:
            # Initialize T,S as linear functions with max depth
            max_bottom_depth = -config.getfloat('vertical_grid',
                                                'bottom_depth')
            frac = (0. - ds.zMid) / (0. - max_bottom_depth)
            ds['temperature'] = \
                (1.0 - frac) * init_top_temp + frac * init_bot_temp
            ds['salinity'] = \
                (1.0 - frac) * init_top_sal + frac * init_bot_sal

        if thin_film_present:
            # for thin film cells, set temperature to freezing point
            # TODO consider setting salinity to 0
            thin_film_mask = np.logical_and(
                mask.values, np.logical_not(floating_mask.values))
            # These coefficients are hard-coded as the defaults in the namelist
            # Note that using the land ice pressure rather than the pressure at
            # floatation will mean that there is a small amount of cooling from
            # grounding line retreat. However, the thin film should be thin
            # enough that this effect isn't signicant.
            freezing_temp = (6.22e-2 +
                             -5.63e-2 * init_bot_sal +
                             -7.43e-8 * land_ice_pressure +
                             -1.74e-10 * land_ice_pressure * init_bot_sal)
            _, thin_film_temp = np.meshgrid(ds.refZMid, freezing_temp)
            _, thin_film_mask = np.meshgrid(ds.refZMid, thin_film_mask)
            thin_film_temp = np.expand_dims(thin_film_temp, axis=0)
            ds['temperature'] = xr.where(thin_film_mask, thin_film_temp,
                                         ds.temperature)

        # compute coriolis
        coriolis_parameter = section.getfloat('coriolis_parameter')

        ds['fCell'] = coriolis_parameter * xr.ones_like(ds.xCell)
        ds['fEdge'] = coriolis_parameter * xr.ones_like(ds.xEdge)
        ds['fVertex'] = coriolis_parameter * xr.ones_like(ds.xVertex)

        normalVelocity = xr.zeros_like(ds.xEdge)
        normalVelocity = normalVelocity.broadcast_like(ds.refBottomDepth)
        normalVelocity = normalVelocity.transpose('nEdges', 'nVertLevels')
        ds['normalVelocity'] = normalVelocity.expand_dims(dim='Time', axis=0)

        write_netcdf(ds, 'initial_state.nc')

        return ds, frac

    def _plot(self, ds):
        """
        Plot several fields from the initial condition
        """
        config = self.config
        min_column_thickness = config.getfloat('isomip_plus',
                                               'min_column_thickness')

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

        totalColThickness = ds.layerThickness.sum(dim='nVertLevels')
        tol = 1e-10
        plotter.plot_horiz_series(ds.landIceMask.expand_dims(
                                  dim='Time', axis=0),
                                  'landIceMask', 'landIceMask',
                                  True)
        plotter.plot_horiz_series(ds.landIceFloatingMask.expand_dims(
                                  dim='Time', axis=0),
                                  'landIceFloatingMask', 'landIceFloatingMask',
                                  True)
        plotter.plot_horiz_series(ds.landIcePressure.expand_dims(
                                  dim='Time', axis=0),
                                  'landIcePressure', 'landIcePressure',
                                  True, vmin=1e5, vmax=1e7, cmap_scale='log')
        plotter.plot_horiz_series(ds.landIceThickness.expand_dims(
                                  dim='Time', axis=0),
                                  'landIceThickness', 'landIceThickness',
                                  True, vmin=0, vmax=1e3,
                                  contour_field=totalColThickness)
        plotter.plot_horiz_series(ds.ssh, 'ssh', 'ssh',
                                  True, vmin=-700, vmax=0)
        bottomDepth = ds.bottomDepth.expand_dims(dim='Time', axis=0)
        plotter.plot_horiz_series(bottomDepth,
                                  'bottomDepth', 'bottomDepth',
                                  True, vmin=0, vmax=700)
        plotter.plot_horiz_series(ds.ssh + bottomDepth,
                                  'H', 'H', True,
                                  vmin=min_column_thickness + tol, vmax=700,
                                  cmap_set_under='k', cmap_scale='log')
        plotter.plot_horiz_series(totalColThickness,
                                  'totalColThickness', 'totalColThickness',
                                  True, vmin=min_column_thickness + 1e-10,
                                  vmax=700, cmap_set_under='r')
        plotter.plot_horiz_series(ds.landIceFraction.expand_dims(
                                  dim='Time', axis=0),
                                  'landIceFraction', 'landIceFraction',
                                  True, vmin=0 + tol, vmax=1 - tol,
                                  cmap='cmo.balance',
                                  cmap_set_under='k', cmap_set_over='r')
        plotter.plot_horiz_series(ds.landIceFloatingFraction.expand_dims(
                                  dim='Time', axis=0),
                                  'landIceFloatingFraction',
                                  'landIceFloatingFraction',
                                  True, vmin=0 + tol, vmax=1 - tol,
                                  cmap='cmo.balance',
                                  cmap_set_under='k', cmap_set_over='r')
        plotter.plot_horiz_series(ds.landIceGroundedFraction.expand_dims(
                                  dim='Time', axis=0),
                                  'landIceGroundedFraction',
                                  'landIceGroundedFraction',
                                  True, vmin=0 + tol, vmax=1 - tol,
                                  cmap='cmo.balance',
                                  cmap_set_under='k', cmap_set_over='r')
        plotter.plot_horiz_series(ds.oceanFracObserved.expand_dims(
                                  dim='Time', axis=0),
                                  'oceanFracObserved', 'oceanFracObserved',
                                  True, vmin=0 + tol, vmax=1 - tol,
                                  cmap='cmo.balance',
                                  cmap_set_under='k', cmap_set_over='r')
        plotter.plot_layer_interfaces()

        plotter.plot_3d_field_top_bot_section(
            ds.layerThickness, nameInTitle='layerThickness',
            prefix='h', units='m',
            vmin=min_column_thickness + tol, vmax=50,
            cmap='cmo.deep_r', cmap_set_under='r')

        plotter.plot_3d_field_top_bot_section(
            ds.zMid, nameInTitle='zMid', prefix='zmid', units='m',
            vmin=-720., vmax=0., cmap='cmo.deep_r')

        plotter.plot_3d_field_top_bot_section(
            ds.temperature, nameInTitle='temperature', prefix='temp',
            units='C', vmin=-2., vmax=1., cmap='cmo.thermal')

        plotter.plot_3d_field_top_bot_section(
            ds.salinity, nameInTitle='salinity', prefix='salin',
            units='PSU', vmin=33.8, vmax=34.7, cmap='cmo.haline')

    def _compute_restoring(self, ds, frac):
        config = self.config
        section = config['isomip_plus']

        ref_density = constants['SHR_CONST_RHOSW']

        ds_forcing = xr.Dataset()

        restore_top_temp = section.getfloat('restore_top_temp')
        restore_bot_temp = section.getfloat('restore_bot_temp')
        restore_top_sal = section.getfloat('restore_top_sal')
        restore_bot_sal = section.getfloat('restore_bot_sal')
        ds_forcing['temperatureInteriorRestoringValue'] = \
            (1.0 - frac) * restore_top_temp + frac * restore_bot_temp
        ds_forcing['salinityInteriorRestoringValue'] = \
            (1.0 - frac) * restore_top_sal + frac * restore_bot_sal

        restore_rate = section.getfloat('restore_rate')
        restore_xmin = section.getfloat('restore_xmin')
        restore_xmax = section.getfloat('restore_xmax')
        frac = np.maximum(
            (ds.xIsomipCell - restore_xmin) / (restore_xmax - restore_xmin),
            0.)
        frac = frac.broadcast_like(
            ds_forcing.temperatureInteriorRestoringValue)

        # convert from 1/days to 1/s
        ds_forcing['temperatureInteriorRestoringRate'] = \
            frac * restore_rate / constants['SHR_CONST_CDAY']
        ds_forcing['salinityInteriorRestoringRate'] = \
            ds_forcing.temperatureInteriorRestoringRate

        # compute "evaporation"
        restore_evap_rate = section.getfloat('restore_evap_rate')

        mask = np.logical_and(ds.xIsomipCell >= restore_xmin,
                              ds.xIsomipCell <= restore_xmax)
        mask = mask.expand_dims(dim='Time', axis=0)
        # convert to m/s, negative for evaporation rather than precipitation
        evap_rate = -restore_evap_rate / (constants['SHR_CONST_CDAY'] * 365)
        # PSU*m/s to kg/m^2/s
        sflux_factor = 1.
        # C*m/s to W/m^2
        hflux_factor = 1. / (ref_density * constants['SHR_CONST_CPSW'])
        ds_forcing['evaporationFlux'] = mask * ref_density * evap_rate
        ds_forcing['seaIceSalinityFlux'] = \
            mask * evap_rate * restore_top_sal / sflux_factor
        ds_forcing['seaIceHeatFlux'] = \
            mask * evap_rate * restore_top_temp / hflux_factor

        x_max = np.max(ds.xIsomipCell.values)
        ds_forcing['tidalInputMask'] = xr.where(
            ds.xIsomipCell > (x_max - 0.6 * self.resolution * 1e3), 1.0,
            0.0)

        write_netcdf(ds_forcing, 'init_mode_forcing_data.nc')

    def _write_time_varying_forcing(self, ds_init, ice_density,
                                    land_ice_pressure_unscaled):
        """
        Write time-varying land-ice forcing
        """

        config = self.config
        ds_forcing = xr.Dataset()
        dates = config.get('isomip_plus_forcing', 'dates')
        dates = [date.ljust(64)
                 for date in dates.replace(',', ' ').split()]
        scales = config.get('isomip_plus_forcing', 'scales')
        scales = [float(scale)
                  for scale in scales.replace(',', ' ').split()]

        # The initial condition already has the first scale value applied
        land_ice_pressure_forcing = ds_init.landIcePressure.copy()
        land_ice_fraction_forcing = ds_init.landIceFraction.copy()
        land_ice_floating_fraction_forcing = \
            ds_init.landIceFloatingFraction.copy()

        # We add additional time slices for the remaining scale values
        for scale in scales[1:]:
            land_ice_pressure_forcing = xr.concat(
                [land_ice_pressure_forcing,
                 scale * land_ice_pressure_unscaled],
                'Time')
            land_ice_fraction_forcing = xr.concat(
                [land_ice_fraction_forcing, ds_init.landIceFraction],
                'Time')
            # Since floating fraction does not change, none of the thin film
            # cases allow for the area undergoing melting to change
            land_ice_floating_fraction_forcing = xr.concat(
                [land_ice_floating_fraction_forcing,
                 ds_init.landIceFloatingFraction],
                'Time')

        ds_forcing['xtime'] = xr.DataArray(data=dates,
                                           dims=('Time')).astype('S')
        ds_forcing['landIcePressureForcing'] = land_ice_pressure_forcing
        ds_forcing.landIcePressureForcing.attrs['units'] = 'm'
        ds_forcing.landIcePressureForcing.attrs['long_name'] = \
            'Pressure from the weight of land ice at the ice-ocean ' \
            'interface'
        ds_forcing['landIceFractionForcing'] = land_ice_fraction_forcing
        ds_forcing.landIceFractionForcing.attrs['long_name'] = \
            'The fraction of each cell covered by land ice'
        ds_forcing['landIceFloatingFractionForcing'] = \
            land_ice_floating_fraction_forcing
        ds_forcing.landIceFloatingFractionForcing.attrs['long_name'] = \
            'The fraction of each cell covered by floating land ice'

        ds_forcing.encoding['unlimited_dims'] = {'Time'}
        # write_netcdf is not used here because it does not yet support
        # multiple time levels
        ds_forcing.to_netcdf('land_ice_forcing.nc')
