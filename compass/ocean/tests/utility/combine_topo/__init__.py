import os
import subprocess
from datetime import datetime

import numpy as np
import pyproj
import xarray as xr
from dask.diagnostics import ProgressBar
from pyremap import ProjectionGridDescriptor

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
    TODO: Document target grid throughout

    Attributes
    ----------
    ...
    """

    def __init__(self, test_case, target_grid):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.utility.combine_topo.CombineTopo
            The test case this step belongs to
        target_grid : `str`, either "lat_lon" or "cubed_sphere"
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

        # Parse resolution
        if self.target_grid == 'cubed_sphere':
            self.resolution = section.getint('resolution')
            self.resolution_name = f'ne{self.resolution}'
        elif self.target_grid == 'lat_lon':
            self.resolution = section.getfloat('resolution')
            self.resolution_name = f'{self.resolution:.4f}_degree'

        # Build combined filename
        combined_filename = '_'.join([
            antarctic_filename.strip('.nc'), global_filename.strip('.nc'),
            self.resolution_name, datetime.now().strftime('%Y%m%d.nc'),
        ])

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
        self.add_output_file(filename=combined_filename)

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
        self._modify_gebco()
        self._modify_bedmachine()
        self._create_gebco_tiles()
        self._create_bedmachine_scrip_file()
        self._create_target_scrip_file()
        self._remap_gebco()
        self._remap_bedmachine()


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
        gebco.to_netcdf(out_filename)
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
        bedmachine['bathymetry'] = bedmachine.bed.where(ocean_mask, 0.)
        bedmachine['thickness'] = bedmachine.thickness.where(ocean_mask, 0.)
        bedmachine['ice_draft'] = \
            (bedmachine.surface - bedmachine.thickness).where(ocean_mask, 0.)
        bedmachine.ice_draft.attrs['units'] = 'meters'
        bedmachine['ice_mask'] = ice_mask
        bedmachine['grounded_mask'] = grounded_mask
        bedmachine['ocean_mask'] = ocean_mask

        # Remove all other variables
        varlist = [
            'bathymetry', 'ice_draft', 'thickness',
            'ice_mask', 'grounded_mask', 'ocean_mask',
        ]
        bedmachine = bedmachine[varlist]

        # Write modified BedMachine to netCDF
        bedmachine.to_netcdf(out_filename)
        logger.info('  Done.')


    def _create_gebco_tile(self, lon_tile, lat_tile):
        """
        Create GEBCO tile
        """
        logger = self.logger

        # Parse config
        config = self.config
        section = config['combine_topo']
        lat_tiles = section.getint('lat_tiles')
        lon_tiles = section.getint('lon_tiles')
        global_name = section.get('global_filename').strip('.nc')

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
        out_filename = f'tiles/{global_name}_tile_{lon_tile}_{lat_tile}.nc'
        write_job = tile.to_netcdf(out_filename, compute=False)
        with ProgressBar():
            logger.info(f'    writing {out_filename}')
            write_job.compute()


    def _create_gebco_tiles(self):
        """
        Create GEBCO tiles. Wrapper around `_create_gebco_tile`
        """
        logger = self.logger
        logger.info('Creating GEBCO tiles')

        # Make tiles directory
        os.makedirs('tiles', exist_ok=True)

        # Parse config
        config = self.config
        section = config['combine_topo']
        lat_tiles = section.getint('lat_tiles')
        lon_tiles = section.getint('lon_tiles')

        # Loop through tiles
        for lat_tile in range(lat_tiles):
            for lon_tile in range(lon_tiles):

                # Create GEBCO tile
                self._create_gebco_tile(lon_tile, lat_tile)

        logger.info('  Done.')


    def _create_bedmachine_scrip_file(self):
        """
        Create SCRIP file for BedMachineAntarctica data
        """
        logger = self.logger
        logger.info('Creating BedMachine SCRIP file')

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

        # Create SCRIP file
        bedmachine_descriptor = ProjectionGridDescriptor.read(
            projection, in_filename, 'BedMachineAntarctica500m',
        )
        bedmachine_descriptor.to_scrip(out_filename)

        logger.info('  Done.')


    def _create_target_scrip_file(self):
        """
        Create SCRIP file for either the x.xxxx degree (lat-lon) mesh or
        the NExxx (cubed-sphere) mesh, depending on the value of `self.target_grid`
        References:
          https://github.com/ClimateGlobalChange/tempestremap
          https://acme-climate.atlassian.net/wiki/spaces/DOC/pages/872579110/
          Running+E3SM+on+New+Atmosphere+Grids
        """
        logger = self.logger
        logger.info('Creating EXODUS and SCRIP files')

        # Create EXODUS file
        if self.target_grid == 'cubed_sphere':
            args = [
                'GenerateCSMesh', '--alt', '--res', f'{self.resolution}',
                '--file', f'{self.resolution_name}.g',
            ]
        elif self.target_grid == 'lat_lon':
            nlon = int(360 / self.resolution)
            nlat = int(nlon / 2)
            args = [
                'GenerateRLLMesh',
                '--lon', f'{nlon}', '--lat', f'{nlat}',
                '--file', f'{self.resolution_name}.g',
            ]
        logger.info(f'    Running: {" ".join(args)}')
        subprocess.run(args, check=True)

        # Create SCRIP file
        args = [
            'ConvertMeshToSCRIP', '--in', f'{self.resolution_name}.g',
            '--out', f'{self.resolution_name}.scrip.nc',
        ]
        logger.info(f'    Running: {" ".join(args)}')
        subprocess.run(args, check=True)

        logger.info('  Done.')


    def _create_weights(self, in_filename, out_filename):
        """
        Create weights file for remapping to `self.target_grid`
        """
        logger = self.logger

        config = self.config
        method = config.get('combine_topo', 'method')

        # Generate weights file
        args = [
            'srun', '-n', f'{self.ntasks}',
            'ESMF_RegridWeightGen',
            '--source', in_filename,
            '--destination', f'{self.resolution_name}.scrip.nc',
            '--weight', out_filename,
            '--method', method,
            '--netcdf4',
            '--src_regional',
            '--ignore_unmapped',
        ]
        logger.info(f'    running: {" ".join(args)}')
        subprocess.run(args, check=True)


    def _remap_to_target(self, in_filename, mapping_filename, out_filename, default_dims=True):
        """
        Remap to `self.target_grid`
        """
        logger = self.logger

        config = self.config
        section = config['combine_topo']
        renorm_thresh = section.getfloat('renorm_thresh')

        # Build command args
        args = [
            'ncremap', '-m', mapping_filename, '--vrb=1',
            f'--renormalize={renorm_thresh}',
        ]

        # Append non-default dimensions, if any
        if not default_dims:
            args = args + ['-R', '--rgr lat_nm=y --rgr lon_nm=x']

        # Append input and output file names
        args = args + [in_filename, out_filename]

        # Remap to target grid
        logger.info(f'    Running: {" ".join(args)}')
        subprocess.run(args, check=True)


    def _remap_gebco(self):
        """
        Remap GEBCO
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

        # Create tile maps and remapped tiles
        for lat_tile in range(lat_tiles):
            for lon_tile in range(lon_tiles):

                # File names
                tile_suffix = f'tile_{lon_tile}_{lat_tile}.nc'
                tile_filename = f'tiles/{global_name}_{tile_suffix}'
                mapping_filename = f'tiles/map_{global_name}_to_{self.resolution_name}_{method}_{tile_suffix}'
                remapped_filename = f'tiles/{global_name}_{self.resolution_name}_{tile_suffix}'

                # Call remapping functions
                self._create_weights(tile_filename, mapping_filename)
                self._remap_to_target(tile_filename, mapping_filename, remapped_filename)

        logger.info('  Done.')


    def _remap_bedmachine(self):
        """
        Remap BedMachineAntarctica
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
        mapping_filename = f'map_{antarctic_name}_to_{self.resolution_name}_{method}.nc'
        remapped_filename = f'{antarctic_name}_{self.resolution_name}.nc'

        # Call remapping functions
        self._create_weights(scrip_filename, mapping_filename)
        self._remap_to_target(
            in_filename, mapping_filename, remapped_filename, default_dims=False,
        )

        logger.info('  Done.')


    def _combine(self):
        """
        """

        combined = xr.Dataset()

        # Combine remapped GEBCO tiles
        tile_prefix = f'tiles/{global_name}_{self.resolution_name}_tile'
        for lat_tile in range(lat_tiles):
            for lon_tile in range(lon_tiles):
                in_filename = f'{tile_prefix}_{lon_tile}_{lat_tile}.nc'
                tile = xr.open_dataset(in_filename)
                bathy = tile.elevation
                mask = bathy.notnull()
                bathy = bathy.where(mask, 0.)
                if 'bathymetry' in combined:
                    combined['bathymetry'] = combined.bathymetry + bathy
                else:
                    combined['bathymetry'] = bathy

        # Open BedMachine and blend into combined bathy with alpha factor
        bedmachine = xr.open_dataset(antarctic_filename)
        alpha = (combined.lat - latmin) / (latmax - latmin)
        alpha = np.maximum(np.minimum(alpha, 1.0), 0.0)
        bathy = bedmachine.bathymetry
        valid = bathy.notnull()
        bathy = bathy.where(valid, 0.)
        combined['bathymetry'] = alpha * combined.bathymetry + (1.0 - alpha) * bathy
        combined['bathymetry'] = combined.bathymetry.where(combined.bathymetry < 0, 0.)

        # Handle remaining variables
        for field in ['ice_draft', 'thickness']:
            combined[field] = bedmachine[field].astype(float)
        for field in ['bathymetry', 'ice_draft', 'thickness']:
            combined[field].attrs['unit'] = 'meters'
        fill = {'ice_mask': 0., 'grounded_mask': 0., 'ocean_mask': combined['bathymetry'] < 0.}
        for field in fill.keys():
            valid = bedmachine[field].notnull()
            combined[field] = bedmachine[field].where(valid, fill[field])
        combined['water_column'] = combined['ice_draft'] - combined['bathymetry']
        combined.water_column.attrs['units'] = 'meters'
        combined = combined.drop_vars(['x', 'y'])

        combined.to_netcdf(combined_filename)
