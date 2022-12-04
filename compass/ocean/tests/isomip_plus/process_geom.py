import xarray as xr
import numpy as np
import scipy.ndimage.filters as filters

from compass.step import Step


class ProcessGeom(Step):
    """
    A step for processing the ISOMIP+ geometry for a given experiment

    Attributes
    ----------
    resolution : float
        The horizontal resolution (km) of the test case

    thin_film_present: bool
        Whether the run includes a thin film below grounded ice
    """
    def __init__(self, test_case, resolution, experiment, thin_film_present):
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
        """
        super().__init__(test_case=test_case, name='process_geom')
        self.resolution = resolution
        self.thin_film_present = thin_film_present

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

        self.add_output_file('input_geometry_processed.nc')

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        thin_film_present = self.thin_film_present

        section = config['isomip_plus']
        filter_sigma = section.getfloat('topo_smoothing')*self.resolution
        min_ice_thickness = section.getfloat('min_ice_thickness')
        draft_scaling = section.getfloat('draft_scaling')
        # km to m
        x_min = 1e3*section.getfloat('x_min')

        # a buffer of cells for masking "land" around the domain
        buffer = 1

        ds_in = xr.open_dataset('input_geometry.nc')
        mask = ds_in.x >= x_min
        ds_in = ds_in.isel(x=mask)

        rename = {'upperSurface': 'Z_ice_surface',
                  'lowerSurface': 'Z_ice_draft',
                  'bedrockTopography': 'Z_bed',
                  'floatingMask': 'floatingIceFraction',
                  'groundedMask': 'landFraction',
                  'openOceanMask': 'openOceanFraction'}

        ds_in = ds_in.rename(rename)

        x_in = ds_in.x.values
        y_in = ds_in.y.values

        dx = x_in[1] - x_in[0]
        dy = y_in[1] - y_in[0]

        nx = ds_in.sizes['x'] + 2 * buffer
        ny = ds_in.sizes['y'] + 2 * buffer

        x_out = x_in[0] + dx * (-buffer + np.arange(nx))
        y_out = y_in[0] + dy * (-buffer + np.arange(ny))

        ds = xr.Dataset()
        ds['x'] = ('x', x_out)
        ds['y'] = ('y', y_out)

        land_values = {'Z_ice_surface': 0.,
                       'Z_ice_draft': 0.,
                       'Z_bed': 0.,
                       'floatingIceFraction': 0.,
                       'landFraction': 1.,
                       'openOceanFraction': 0.}

        for var, land_value in land_values.items():
            out_field = land_value * np.ones((ny, nx), float)
            out_field[buffer:-buffer, buffer:-buffer] = ds_in[var].values
            ds[var] = (('y', 'x'), out_field)

        ds['iceThickness'] = ds.Z_ice_surface - ds.Z_ice_draft

        ds['Z_ice_draft'] = draft_scaling*ds.Z_ice_draft

        # take care of calving criterion
        mask = np.logical_or(ds.floatingIceFraction <= 0.1,
                             ds.iceThickness >= min_ice_thickness)

        for var in ['Z_ice_surface', 'Z_ice_draft', 'floatingIceFraction']:
            ds[var] = xr.where(mask, ds[var], 0.0)
        ds['openOceanFraction'] = xr.where(mask, ds.openOceanFraction,
                                           1. - ds.landFraction)

        if thin_film_present:
            smooth_mask = xr.where(ds.Z_bed >= 0,
                                   ds.landFraction, 0.)
        else:
            smooth_mask = ds.landFraction

        self._smooth_geometry(smooth_mask, ds, filter_sigma)

        # copy attributes
        for var in ['x', 'y', 'Z_ice_surface', 'Z_ice_draft', 'Z_bed',
                    'floatingIceFraction', 'landFraction',
                    'openOceanFraction']:
            attrs = ds_in[var].attrs
            if 'units' in attrs and attrs['units'] == 'unitless':
                attrs.pop('units')
            attrs.pop('_FillValue', None)
            ds[var].attrs = attrs

        ds.smoothedDraftMask.attrs['description'] = \
            'floating fraction after smoothing'

        ds.to_netcdf('input_geometry_processed.nc')

    @staticmethod
    def _smooth_geometry(land_fraction, ds, filter_sigma, threshold=0.01):
        """
        Smoothing is performed using only the topography in the portion of the
        grid that is ocean. This prevents the kink in the ice draft across the
        grounding line or regions of bare bedrock from influencing the smoothed
        topography.

        The calving front is smoothed as well because MPAS-O does not support
        a sheer calving face
        """
        ocean_fraction = 1. - land_fraction.values
        smoothed_mask = filters.gaussian_filter(ocean_fraction, filter_sigma,
                                                mode='constant', cval=0.)
        mask = smoothed_mask > threshold

        draft = filters.gaussian_filter(ds.Z_ice_draft.values * ocean_fraction,
                                        filter_sigma, mode='constant', cval=0.)
        draft[mask] /= smoothed_mask[mask]

        bed = filters.gaussian_filter(ds.Z_bed * ocean_fraction, filter_sigma,
                                      mode='constant', cval=0.)
        bed[mask] /= smoothed_mask[mask]

        smoothed_draft_mask = filters.gaussian_filter(ds.floatingIceFraction,
                                                      filter_sigma,
                                                      mode='constant', cval=0.)
        smoothed_draft_mask[mask] /= smoothed_mask[mask]

        ds['Z_ice_draft'] = (('y', 'x'), draft)
        ds['Z_bed'] = (('y', 'x'), bed)
        ds['smoothedDraftMask'] = (('y', 'x'), smoothed_draft_mask)
