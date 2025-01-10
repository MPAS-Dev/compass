import os
import pathlib
from datetime import datetime
from glob import glob

import netCDF4
import numpy as np
import pyproj
import xarray as xr
from mpas_tools.logging import check_call
from pyremap import ProjectionGridDescriptor, get_lat_lon_descriptor

from compass.parallel import run_command
from compass.step import Step
from compass.testcase import TestCase


class CombineTopo(TestCase):
    """
    A test case for combining GEBCO 2023 with BedMachineAntarctica topography
    datasets
    """

    def __init__(self, test_group, target_grid):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.utility.Utility
            The test group that this test case belongs to
        target_grid : str
            either "lat_lon" or "cubed_sphere"
        """

        subdir = os.path.join('combine_topo', target_grid)

        super().__init__(
            test_group=test_group, name='combine_topo', subdir=subdir,
        )

        self.add_step(Combine(test_case=self, target_grid=target_grid))


class Combine(Step):
    """
    A step for combining GEBCO 2023 with BedMachineAntarctica topography
    datasets

    Attributes
    ----------
    target_grid : str
        either "lat_lon" or "cubed_sphere"
    resolution : float
        degrees (float) or face subdivisions (int)
    resolution_name: str
        either x.xxxx_degrees or NExxx
    """

    def __init__(self, test_case, target_grid):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.utility.combine_topo.CombineTopo
            The test case this step belongs to
        target_grid : str
            either "lat_lon" or "cubed_sphere"
        """
        super().__init__(
            test_case, name='combine', ntasks=None, min_tasks=None,
        )
        self.target_grid = target_grid
        self.resolution = None
        self.resolution_name = None

    def setup(self):
        """
        Set up the step in the work directory, including downloading any
        dependencies.
        """
        super().setup()

        config = self.config
        section = config['combine_topo']

        # Get input filenames and resolution
        antarctic_filename = section.get('antarctic_filename')
        global_filename = section.get('global_filename')

        # Add bathymetry data input files
        self.add_input_file(
            filename=antarctic_filename,
            target=antarctic_filename,
            database='bathymetry_database',
        )
        self.add_input_file(
            filename=global_filename,
            target=global_filename,
            database='bathymetry_database',
        )
        self._set_res_and_outputs(update=False)

        # Get ntasks and min_tasks
        self.ntasks = section.getint('ntasks')
        self.min_tasks = section.getint('min_tasks')

    def constrain_resources(self, available_resources):
        """
        Constrain ``cpus_per_task`` and ``ntasks`` based on the number of
        cores available to this step

        Parameters
        ----------
        available_resources : dict
            The total number of cores available to the step
        """
        config = self.config
        self.ntasks = config.getint('combine_topo', 'ntasks')
        self.min_tasks = config.getint('combine_topo', 'min_tasks')
        super().constrain_resources(available_resources)

    def run(self):
        """
        Run this step of the test case
        """
        self._set_res_and_outputs(update=True)
        self._modify_gebco()
        self._modify_bedmachine()
        self._create_target_scrip_file()
        self._remap_gebco()
        self._remap_bedmachine()
        self._combine()
        self._cleanup()

    def _set_res_and_outputs(self, update):
        """
        Set or update the resolution and output filenames based on config
        options
        """
        config = self.config
        section = config['combine_topo']

        # Get input filenames and resolution
        antarctic_filename = section.get('antarctic_filename')
        global_filename = section.get('global_filename')
        # Parse resolution and update resolution attributes
        if self.target_grid == 'cubed_sphere':
            resolution = section.getint('resolution_cubedsphere')
            if update and resolution == self.resolution:
                # nothing to do
                return
            self.resolution = resolution
            self.resolution_name = f'ne{resolution}'
        elif self.target_grid == 'lat_lon':
            resolution = section.getfloat('resolution_latlon')
            if update and resolution == self.resolution:
                # nothing to do
                return
            self.resolution = resolution
            self.resolution_name = f'{resolution:.4f}_degree'

        # Start over with empty outputs
        self.outputs = []

        # Build output filenames
        datestamp = datetime.now().strftime('%Y%m%d')
        scrip_filename = f'{self.resolution_name}_{datestamp}.scrip.nc'
        combined_filename = '_'.join([
            antarctic_filename.strip('.nc'),
            global_filename.strip('.nc'),
            self.resolution_name, f'{datestamp}.nc',
        ])
        self.add_output_file(filename=scrip_filename)
        self.add_output_file(filename=combined_filename)

        if update:
            # We need to set absolute paths
            step_dir = self.work_dir
            self.outputs = [os.path.abspath(os.path.join(step_dir, filename))
                            for filename in self.outputs]

    def _modify_gebco(self):
        """
        Modify GEBCO to include lon/lat bounds located at grid edges
        """
        logger = self.logger
        logger.info('Adding bounds to GEBCO lat/lon')

        # Parse config
        config = self.config
        in_filename = config.get('combine_topo', 'global_filename')
        out_filename = in_filename.replace('.nc', '_cf.nc')

        # Modify GEBCO
        gebco = xr.open_dataset(in_filename)
        lat = gebco.lat
        lon = gebco.lon
        dlat = lat.isel(lat=1) - lat.isel(lat=0)
        dlon = lon.isel(lon=1) - lon.isel(lon=0)
        lat_bnds = xr.concat([lat - 0.5 * dlat, lat + 0.5 * dlat], dim='bnds')
        lon_bnds = xr.concat([lon - 0.5 * dlon, lon + 0.5 * dlon], dim='bnds')
        gebco['lat_bnds'] = lat_bnds.transpose('lat', 'bnds')
        gebco['lon_bnds'] = lon_bnds.transpose('lon', 'bnds')
        gebco.lat.attrs['bounds'] = 'lat_bnds'
        gebco.lon.attrs['bounds'] = 'lon_bnds'

        # Write modified GEBCO to netCDF
        _write_netcdf_with_fill_values(gebco, out_filename)
        logger.info('  Done.')

    def _modify_bedmachine(self):
        """
        Modify BedMachineAntarctica to compute the fields needed by MPAS-Ocean
        """
        logger = self.logger
        logger.info('Modifying BedMachineAntarctica with MPAS-Ocean names')

        # Parse config
        config = self.config
        in_filename = config.get('combine_topo', 'antarctic_filename')
        out_filename = in_filename.replace('.nc', '_mod.nc')

        # Load BedMachine and get ice, ocean and grounded masks
        bedmachine = xr.open_dataset(in_filename)
        mask = bedmachine.mask
        ice_mask = (mask != 0).astype(float)
        ocean_mask = np.logical_or(mask == 0, mask == 3).astype(float)
        grounded_mask = np.logical_or(np.logical_or(mask == 1, mask == 2),
                                      mask == 4).astype(float)

        # Add new variables and apply ocean mask
        bedmachine['bathymetry'] = bedmachine.bed
        bedmachine['thickness'] = bedmachine.thickness
        bedmachine['ice_draft'] = bedmachine.surface - bedmachine.thickness
        bedmachine.ice_draft.attrs['units'] = 'meters'
        bedmachine['ice_mask'] = ice_mask
        bedmachine['grounded_mask'] = grounded_mask
        bedmachine['ocean_mask'] = ocean_mask
        bedmachine['valid_mask'] = xr.ones_like(ocean_mask)

        # Remove all other variables
        varlist = [
            'bathymetry', 'ice_draft', 'thickness',
            'ice_mask', 'grounded_mask', 'ocean_mask', 'valid_mask'
        ]
        bedmachine = bedmachine[varlist]

        # Write modified BedMachine to netCDF
        _write_netcdf_with_fill_values(bedmachine, out_filename)
        logger.info('  Done.')

    def _create_gebco_tile(self, lon_tile, lat_tile):
        """
        Create GEBCO tile

        Parameters
        ----------
        lon_tile : int
            tile number along lon dim
        lat_tile : int
            tile number along lat dim
        """
        logger = self.logger

        # Parse config
        config = self.config
        section = config['combine_topo']
        lat_tiles = section.getint('lat_tiles')
        lon_tiles = section.getint('lon_tiles')
        global_name = section.get('global_filename').strip('.nc')
        out_filename = f'tiles/{global_name}_tile_{lon_tile}_{lat_tile}.nc'

        logger.info(f'    creating {out_filename}')

        # Load GEBCO
        gebco = xr.open_dataset(f'{global_name}_cf.nc')

        # Build lat and lon arrays for tile
        nlat = gebco.sizes['lat']
        nlon = gebco.sizes['lon']
        nlat_tile = nlat // lat_tiles
        nlon_tile = nlon // lon_tiles

        # Build tile latlon indices
        lat_indices = [lat_tile * nlat_tile, (lat_tile + 1) * nlat_tile]
        lon_indices = [lon_tile * nlon_tile, (lon_tile + 1) * nlon_tile]
        if lat_tile == lat_tiles - 1:
            lat_indices[1] = max([lat_indices[1], nlat])
        else:
            lat_indices[1] += 1
        if lon_tile == lon_tiles - 1:
            lon_indices[1] = max([lon_indices[1], nlon])
        else:
            lon_indices[1] += 1
        lat_indices = np.arange(*lat_indices)
        lon_indices = np.arange(*lon_indices)

        # Duplicate top and bottom rows to account for poles
        if lat_tile == 0:
            lat_indices = np.insert(lat_indices, 0, 0)
        if lat_tile == lat_tiles - 1:
            lat_indices = np.append(lat_indices, lat_indices[-1])

        # Select tile from GEBCO
        tile = gebco.isel(lat=lat_indices, lon=lon_indices)
        if lat_tile == 0:
            tile.lat.values[0] = -90.  # Correct south pole
        if lat_tile == lat_tiles - 1:
            tile.lat.values[-1] = 90.  # Correct north pole

        # Write tile to netCDF
        _write_netcdf_with_fill_values(tile, out_filename)

    def _create_bedmachine_scrip_file(self):
        """
        Create SCRIP file for BedMachineAntarctica data using pyremap
        """
        logger = self.logger
        logger.info('    Creating BedMachine SCRIP file')

        # Parse config
        config = self.config
        section = config['combine_topo']
        antarctic_name = section.get('antarctic_filename').strip('.nc')
        in_filename = f'{antarctic_name}_mod.nc'
        out_filename = f'{antarctic_name}.scrip.nc'

        # Define projection
        projection = pyproj.Proj(
            '+proj=stere +lat_ts=-71.0 +lat_0=-90 +lon_0=0.0 '
            '+k_0=1.0 +x_0=0.0 +y_0=0.0 +ellps=WGS84'
        )

        # Create SCRIP file using pyremap
        bedmachine_descriptor = ProjectionGridDescriptor.read(
            projection, in_filename, 'BedMachineAntarctica500m',
        )
        bedmachine_descriptor.format = 'NETCDF3_64BIT_DATA'
        bedmachine_descriptor.to_scrip(out_filename)

    def _create_target_scrip_file(self):
        """
        Create SCRIP file for either the x.xxxx degree (lat-lon) mesh or the
        NExxx (cubed-sphere) mesh, depending on the value of `self.target_grid`
        References:
          https://acme-climate.atlassian.net/wiki/spaces/DOC/pages/872579110/
          Running+E3SM+on+New+Atmosphere+Grids
        """
        logger = self.logger
        logger.info(f'Create SCRIP file for {self.resolution_name} mesh')

        out_filename = self.outputs[0]
        stem = pathlib.Path(out_filename).stem
        netcdf4_filename = f'{stem}.netcdf4.nc'

        # Build cubed sphere SCRIP file using tempestremap
        if self.target_grid == 'cubed_sphere':

            # Create EXODUS file
            args = [
                'GenerateCSMesh', '--alt', '--res', f'{self.resolution}',
                '--file', f'{self.resolution_name}.g',
            ]
            check_call(args, logger)

            # Create SCRIP file
            args = [
                'ConvertMeshToSCRIP', '--in', f'{self.resolution_name}.g',
                '--out', netcdf4_filename,
            ]
            check_call(args, logger)

            # ConvertMeshToSCRIP doesn't support NETCDF3_64BIT_DATA, so use
            # ncks
            args = [
                'ncks', '-O', '-5',
                netcdf4_filename,
                out_filename,
            ]
            check_call(args, logger)

        # Build lat-lon SCRIP file using pyremap
        elif self.target_grid == 'lat_lon':
            descriptor = get_lat_lon_descriptor(
                dLon=self.resolution, dLat=self.resolution,
            )
            descriptor.format = 'NETCDF3_64BIT_DATA'
            descriptor.to_scrip(out_filename)

        logger.info('  Done.')

    def _create_weights(self, in_filename, out_filename):
        """
        Create weights file for remapping to `self.target_grid`. Filenames
        are passed as parameters so that the function can be applied to
        GEBCO and BedMachine

        Parameters
        ----------
        in_filename : str
            source file name
        out_filename : str
            weights file name
        """
        config = self.config
        method = config.get('combine_topo', 'method')

        # Generate weights file
        args = [
            'ESMF_RegridWeightGen',
            '--source', in_filename,
            '--destination', self.outputs[0],
            '--weight', out_filename,
            '--method', method,
            '--netcdf4',
            '--src_regional',
            '--ignore_unmapped',
        ]
        run_command(
            args, self.cpus_per_task, self.ntasks,
            self.openmp_threads, config, self.logger,
        )

    def _remap_to_target(
        self, in_filename, mapping_filename, out_filename, default_dims=True,
    ):
        """
        Remap to `self.target_grid`. Filenames are passed as parameters so
        that the function can be applied to GEBCO and BedMachine.

        Parameters
        ----------
        in_filename : str
            source file name
        mapping_filename : str
            weights file name
        out_filename : str
            remapped file name
        default_dims : bool
            default True, if False specify non-default source dims y,x
        """
        # Build command args
        args = [
            'ncremap',
            '-m', mapping_filename,
            '--vrb=1'
        ]

        # Add non-default gridding args
        regridArgs = []
        if not default_dims:
            regridArgs.extend([
                '--rgr lat_nm=y',
                '--rgr lon_nm=x',
            ])
        if self.target_grid == 'lat_lon':
            regridArgs.extend([
                '--rgr lat_nm_out=lat',
                '--rgr lon_nm_out=lon',
                '--rgr lat_dmn_nm=lat',
                '--rgr lon_dmn_nm=lon',
            ])
        if len(regridArgs) > 0:
            args.extend(['-R', ' '.join(regridArgs)])

        # Append input and output file names
        args.extend([in_filename, out_filename])

        # Remap to target grid
        check_call(args, self.logger)

    def _remap_gebco(self):
        """
        Remap GEBCO to `self.target_grid`
        """
        logger = self.logger
        logger.info('Remapping GEBCO data')

        # Parse config
        config = self.config
        section = config['combine_topo']
        global_name = section.get('global_filename').strip('.nc')
        method = section.get('method')
        lat_tiles = section.getint('lat_tiles')
        lon_tiles = section.getint('lon_tiles')

        # Make tiles directory
        os.makedirs('tiles', exist_ok=True)

        # Initialize combined xarray.Dataset
        gebco_remapped = xr.Dataset()

        # Create tile maps and remapped tiles
        for lat_tile in range(lat_tiles):
            for lon_tile in range(lon_tiles):

                # File names
                tile_suffix = f'tile_{lon_tile}_{lat_tile}.nc'
                tile_filename = f'tiles/{global_name}_{tile_suffix}'
                mapping_filename = (
                    f'tiles/map_{global_name}_to_{self.resolution_name}'
                    f'_{method}_{tile_suffix}'
                )
                remapped_filename = (
                    f'tiles/{global_name}_{self.resolution_name}_{tile_suffix}'
                )

                # Call remapping functions
                self._create_gebco_tile(lon_tile, lat_tile)
                self._create_weights(tile_filename, mapping_filename)
                self._remap_to_target(
                    tile_filename, mapping_filename, remapped_filename,
                )

                # Add tile to remapped global bathymetry
                logger.info(f'    adding {remapped_filename}')
                bathy = xr.open_dataset(remapped_filename).elevation
                bathy = bathy.where(bathy.notnull(), 0.)
                if 'bathymetry' in gebco_remapped:
                    gebco_remapped['bathymetry'] = (
                        gebco_remapped.bathymetry + bathy
                    )
                else:
                    gebco_remapped['bathymetry'] = bathy

        # Write tile to netCDF
        out_filename = f'{global_name}_{self.resolution_name}.nc'
        logger.info(f'    writing {out_filename}')
        _write_netcdf_with_fill_values(gebco_remapped, out_filename)

        logger.info('  Done.')

    def _remap_bedmachine(self):
        """
        Remap BedMachineAntarctica to `self.target_grid`
        """
        logger = self.logger
        logger.info('Remapping BedMachine data')

        # Parse config
        config = self.config
        section = config['combine_topo']
        antarctic_name = section.get('antarctic_filename').strip('.nc')
        method = section.get('method')

        # File names
        in_filename = f'{antarctic_name}_mod.nc'
        scrip_filename = f'{antarctic_name}.scrip.nc'
        mapping_filename = (
            f'map_{antarctic_name}_to_{self.resolution_name}_{method}.nc'
        )
        remapped_filename = f'{antarctic_name}_{self.resolution_name}.nc'

        # Call remapping functions
        self._create_bedmachine_scrip_file()
        self._create_weights(scrip_filename, mapping_filename)
        self._remap_to_target(
            in_filename, mapping_filename, remapped_filename,
            default_dims=False,
        )

        logger.info('  Done.')

    def _combine(self):
        """
        Combine remapped GEBCO and BedMachine data sets
        """
        logger = self.logger
        logger.info('Combine BedMachineAntarctica and GEBCO')

        config = self.config
        section = config['combine_topo']
        renorm_thresh = section.getfloat('renorm_thresh')

        # Parse config
        config = self.config
        section = config['combine_topo']
        sfx = f'_{self.resolution_name}.nc'
        global_fname = section.get('global_filename').replace('.nc', sfx)
        antarctic_fname = section.get('antarctic_filename').replace('.nc', sfx)
        latmin = section.getfloat('latmin')
        latmax = section.getfloat('latmax')

        # Load and mask GEBCO
        gebco = xr.open_dataset(global_fname)
        gebco_bathy = gebco.bathymetry
        gebco_bathy = gebco_bathy.where(gebco_bathy.notnull(), 0.)
        gebco_bathy = gebco_bathy.where(gebco_bathy < 0., 0.)

        # Load and mask BedMachine
        bedmachine = xr.open_dataset(antarctic_fname)

        # renormalize variables
        denom = bedmachine.valid_mask
        renorm_mask = denom >= renorm_thresh
        denom = xr.where(renorm_mask, denom, 1.0)
        vars = ['bathymetry', 'thickness', 'ice_draft', 'ice_mask',
                'grounded_mask', 'ocean_mask']

        for var in vars:
            attrs = bedmachine[var].attrs
            bedmachine[var] = (bedmachine[var] / denom).where(renorm_mask)
            bedmachine[var].attrs = attrs

        bed_bathy = bedmachine.bathymetry
        bed_bathy = bed_bathy.where(bed_bathy.notnull(), 0.)
        bed_bathy = bed_bathy.where(bed_bathy < 0., 0.)

        # Blend data sets into combined data set
        combined = xr.Dataset()
        alpha = (gebco.lat - latmin) / (latmax - latmin)
        alpha = np.maximum(np.minimum(alpha, 1.0), 0.0)
        combined['bathymetry'] = (
            alpha * gebco_bathy + (1.0 - alpha) * bed_bathy
        )
        bathy_mask = xr.where(combined.bathymetry < 0., 1.0, 0.0)

        # Add remaining Bedmachine variables to combined Dataset
        for field in ['ice_draft', 'thickness']:
            combined[field] = bathy_mask * bedmachine[field]
        combined['water_column'] = combined.ice_draft - combined.bathymetry
        for field in ['bathymetry', 'ice_draft', 'thickness', 'water_column']:
            combined[field].attrs['unit'] = 'meters'

        # Add masks
        combined['bathymetry_mask'] = bathy_mask
        for field in ['ice_mask', 'grounded_mask', 'ocean_mask']:
            combined[field] = bedmachine[field]

        # Add fill values
        fill_vals = {
            'ice_draft': 0.,
            'thickness': 0.,
            'ice_mask': 0.,
            'grounded_mask': 0.,
            'ocean_mask': bathy_mask,
        }
        for field, fill_val in fill_vals.items():
            valid = combined[field].notnull()
            combined[field] = combined[field].where(valid, fill_val)

        # Save combined bathy to NetCDF
        _write_netcdf_with_fill_values(combined, self.outputs[1])

        logger.info('  Done.')

    def _cleanup(self):
        """
        Clean up work directory
        """
        logger = self.logger
        logger.info('Cleaning up work directory')

        # Remove PETxxx.RegridWeightGen.Log files
        for f in glob('*.RegridWeightGen.Log'):
            os.remove(f)

        logger.info('  Done.')


def _write_netcdf_with_fill_values(ds, filename):
    """ Write an xarray Dataset with NetCDF4 fill values where needed """
    fill_values = netCDF4.default_fillvals
    encoding = {}
    vars = list(ds.data_vars.keys()) + list(ds.coords.keys())
    for var_name in vars:
        # If there's already a fill value attribute, drop it
        ds[var_name].attrs.pop("_FillValue", None)
        is_numeric = np.issubdtype(ds[var_name].dtype, np.number)
        if is_numeric:
            dtype = ds[var_name].dtype
            for fill_type in fill_values:
                if dtype == np.dtype(fill_type):
                    encoding[var_name] = {"_FillValue": fill_values[fill_type]}
                    break
        else:
            encoding[var_name] = {"_FillValue": None}
    ds.to_netcdf(filename, encoding=encoding)
