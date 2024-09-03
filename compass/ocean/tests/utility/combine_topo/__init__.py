import os
import subprocess
from datetime import datetime

import numpy as np
import progressbar
import pyproj
import xarray as xr
from pyremap import LatLonGridDescriptor, ProjectionGridDescriptor, Remapper

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
        date_stamp : `datetime.datetime`, date when test was run
        """
        super().__init__(
            test_case, name='combine', ntasks=None, min_tasks=None,
        )
        self.target_grid = target_grid
        self.date_stamp = None

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
        resolution = section.getint('resolution')

        # Datestamp
        self.datestamp = datetime.now().strftime('%Y%m%d')

        # Build output filename
        combined_filename = _get_combined_filename(
            self.target_grid, self.datestamp, resolution,
            antarctic_filename, global_filename,
        )

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

        # Make tiles directory
        os.makedirs('tiles', exist_ok=True)

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
        self._modify_bedmachine()
        self._modify_gebco()
        self._tile_gebco()
        self._create_ne_scrip()
        self._create_bedmachine_scrip()
        self._get_map_gebco()
        self._get_map_bedmachine()
        self._remap_gebco()
        self._remap_bedmachine()
        self._combine()

    def _modify_bedmachine(self):
        """
        Modify BedMachineAntarctica to compute the fields needed by MPAS-Ocean
        """
        logger = self.logger
        logger.info('Modifying BedMachineAntarctica with MPAS-Ocean names')

        config = self.config
        in_filename = config.get('combine_topo', 'antarctic_filename')
        out_filename = in_filename.replace('.nc', '_mod.nc')

        bedmachine = xr.open_dataset(in_filename)
        mask = bedmachine.mask
        ice_mask = (mask != 0).astype(float)
        ocean_mask = (np.logical_or(mask == 0, mask == 3)).astype(float)
        grounded_mask = np.logical_or(np.logical_or(mask == 1, mask == 2),
                                      mask == 4).astype(float)

        bedmachine['bathymetry'] = bedmachine.bed
        bedmachine['ice_draft'] = bedmachine.surface - bedmachine.thickness
        bedmachine.ice_draft.attrs['units'] = 'meters'
        bedmachine['thickness'] = bedmachine.thickness

        bedmachine['bathymetry'] = bedmachine['bathymetry'].where(ocean_mask, 0.)
        bedmachine['ice_draft'] = bedmachine['ice_draft'].where(ocean_mask, 0.)
        bedmachine['thickness'] = bedmachine['thickness'].where(ocean_mask, 0.)

        bedmachine['ice_mask'] = ice_mask
        bedmachine['grounded_mask'] = grounded_mask
        bedmachine['ocean_mask'] = ocean_mask

        varlist = [
            'bathymetry', 'ice_draft', 'thickness',
            'ice_mask', 'grounded_mask', 'ocean_mask',
        ]

        bedmachine = bedmachine[varlist]

        bedmachine.to_netcdf(out_filename)
        logger.info('  Done.')

    def _modify_gebco(self):
        """
        Modify GEBCO to include lon/lat bounds located at grid edges
        """
        logger = self.logger
        logger.info('Adding bounds to GEBCO lat/lon')

        config = self.config
        in_filename = config.get('combine_topo', 'global_filename')
        out_filename = in_filename.replace('.nc', '_cf.nc')

        gebco = xr.open_dataset(in_filename)
        lat = gebco.lat
        lon = gebco.lon
        dlat = gebco.lat.isel(lat=1) - gebco.lat.isel(lat=0)
        dlon = gebco.lon.isel(lon=1) - gebco.lon.isel(lon=0)
        lat_bnds = xr.concat([lat - 0.5 * dlat, lat + 0.5 * dlat], dim='bnds')
        lon_bnds = xr.concat([lon - 0.5 * dlon, lon + 0.5 * dlon], dim='bnds')
        gebco['lat_bnds'] = lat_bnds.transpose('lat', 'bnds')
        gebco['lon_bnds'] = lon_bnds.transpose('lon', 'bnds')
        gebco.lat.attrs['bounds'] = 'lat_bnds'
        gebco.lon.attrs['bounds'] = 'lon_bnds'

        gebco.to_netcdf(out_filename)
        logger.info('  Done.')

    def _tile_gebco(self):
        """
        Tile GEBCO
        """
        logger = self.logger
        logger.info('Tiling GEBCO data')

        config = self.config
        section = config['combine_topo']
        global_filename = section.get('global_filename')
        lat_tiles = section.getint('lat_tiles')
        lon_tiles = section.getint('lon_tiles')

        in_filename = global_filename.replace('.nc', '_cf.nc')

        gebco = xr.open_dataset(in_filename)
        gebco = gebco.chunk({'lat': lat_tiles * lon_tiles})
        nlat = gebco.sizes['lat']
        nlon = gebco.sizes['lon']
        nlat_tile = nlat // lat_tiles
        nlon_tile = nlon // lon_tiles
        for lat_tile in range(lat_tiles):
            for lon_tile in range(lon_tiles):
                src_filename, _, _ = _get_tile_filenames(global_filename, resolution, method, lat_tile, lon_tile)

                lat_indices = np.arange(
                    lat_tile * nlat_tile - 1, (lat_tile + 1) * nlat_tile + 1
                )
                lon_indices = np.arange(
                    lon_tile * nlon_tile, (lon_tile + 1) * nlon_tile + 1
                )
                lat_indices = np.minimum(np.maximum(lat_indices, 0), nlat - 1)
                lon_indices = np.mod(lon_indices, nlon)
                ds_local = gebco.isel(lat=lat_indices, lon=lon_indices)
                if lat_tile == 0:
                    # need to handle south pole
                    ds_local.lat.values[0] = -90.
                if lat_tile == lat_tiles - 1:
                    # need to handle north pole
                    ds_local.lat.values[-1] = 90.

                write_job = ds_local.to_netcdf(out_filename, compute=False)
                with ProgressBar():
                    logger.info(f'   writing {out_filename}')
                    write_job.compute()

        logger.info('  Done.')

    def _create_ne_scrip(self):
        """
        Create SCRIP file for the NExxx (cubed-sphere) mesh
        Reference:
          https://acme-climate.atlassian.net/wiki/spaces/DOC/pages/872579110/
          Running+E3SM+on+New+Atmosphere+Grids
        """
        logger = self.logger
        logger.info('Creating NE scrip file')

        resolution = self.resolution

        args = [
            'GenerateCSMesh', '--alt', '--res', f'{resolution}',
            '--file', f'ne{resolution}.g',
        ]
        logger.info(f'    Running: {" ".join(args)}')
        subprocess.check_call(args)

        args = [
            'ConvertMeshToSCRIP', '--in', f'ne{resolution}.g',
            '--out', f'ne{resolution}.scrip.nc',
        ]
        logger.info(f'    Running: {" ".join(args)}')
        subprocess.check_call(args)

        logger.info('  Done.')

    def _create_bedmachine_scrip(self):
        """
        Create SCRIP file for the BedMachineAntarctica-v3 bathymetry
        """
        logger = self.logger
        logger.info('Creating Bedmachine scrip file')

        in_filename = self.antarctic_filename
        out_filename = in_filename.replace('.nc', '.scrip.nc')

        projection = pyproj.Proj(
            '+proj=stere +lat_ts=-71.0 +lat_0=-90 +lon_0=0.0 '
            '+k_0=1.0 +x_0=0.0 +y_0=0.0 +ellps=WGS84'
        )

        bedmachine_descriptor = ProjectionGridDescriptor.read(
            projection, in_filename, 'BedMachineAntarctica500m',
        )
        bedmachine_descriptor.to_scrip(out_filename)

        logger.info('  Done.')

    def _get_map_gebco(self):
        """
        Create mapping files for GEBCO tiles onto NE grid
        """
        logger = self.logger
        logger.info('Creating mapping files for GEBCO tiles')

        config = self.config
        section = config['combine_topo']
        method = section.get('method')
        lat_tiles = section.getint('lat_tiles')
        lon_tiles = section.getint('lon_tiles')
        resolution = self.resolution

        global_name = self.global_filename.strip('.nc')
        tile_prefix = f'tiles/{global_name}_tile'
        map_prefix = f'tiles/map_{global_name}_to_ne{resolution}_{method}_tile'
        ne_scrip_filename = f'ne{resolution}.scrip.nc'

        for lat_tile in range(self.lat_tiles):
            for lon_tile in range(self.lon_tiles):
                in_filename = f'{tile_prefix}_{lon_tile}_{lat_tile}.nc'
                out_filename = f'{map_prefix}_{lon_tile}_{lat_tile}.nc'

                args = [
                    'srun', '-n', f'{self.ntasks}',
                    'ESMF_RegridWeightGen',
                    '--source', in_filename,
                    '--destination', ne_scrip_filename,
                    '--weight', out_filename,
                    '--method', method,
                    '--netcdf4',
                    '--src_regional',
                    '--ignore_unmapped',
                ]
                logger.info(f'    Running: {" ".join(args)}')
                subprocess.check_call(args)

        logger.info('  Done.')

    def _get_map_bedmachine(self):
        """
        Create BedMachine to NE mapping file
        """
        logger = self.logger
        logger.info('Creating BedMachine mapping file')

        config = self.config
        method = config.get('combine_topo', 'method')
        resolution = self.resolution
        antarctic_name = self.antarctic_filename.strip('.nc')
        in_filename = f'{antarctic_name}.scrip.nc'
        out_filename = f'map_{antarctic_name}_to_ne{resolution}_{method}.nc'
        ne_scrip_filename = f'ne{resolution}.scrip.nc'

        args = [
            'srun', '-n', f'{self.ntasks}',
            'ESMF_RegridWeightGen',
            '--source', in_filename,
            '--destination', ne_scrip_filename,
            '--weight', out_filename,
            '--method', method,
            '--netcdf4',
            '--src_regional'
            '--ignore_unmapped'
        ]
        logger.info(f'    Running: {" ".join(args)}')
        subprocess.check_call(args)

        logger.info('  Done.')

    def _remap_gebco(self):
        """
        Remap GEBCO tiles to NE grid
        """
        logger = self.logger
        logger.info('Remapping GEBCO tiles to NE grid')

        global_name = self.global_filename.strip('.nc')
        tile_prefix = f'tiles/{global_name}_tile'
        ne_prefix = f'tiles/{global_name}_on_ne_tile'
        map_prefix = f'tiles/map_{global_name}_to_ne{resolution}_{method}_tile'

        for lat_tile in range(self.lat_tiles):
            for lon_tile in range(self.lon_tiles):
                in_filename = f'{tile_prefix}_{lon_tile}_{lat_tile}.nc'
                out_filename = f'{ne_prefix}_{lon_tile}_{lat_tile}.nc'
                mapping_filename = f'{map_prefix}_{lon_tile}_{lat_tile}.nc'

                args = [
                    'ncremap',
                    '-m', mapping_filename,
                    '--vrb=1',
                    f'--renormalize={self.renorm_thresh}',
                    in_filename,
                    out_filename,
                ]
                logger.info(f'    Running: {" ".join(args)}')
                subprocess.check_call(args)

        logger.info('  Done.')

    def _remap_bedmachine(self):
        """
        Remap BedMachine to NE grid
        Still need:
          bedmachine_filename, bedmachine_on_ne_filename,
          mapping_filename, renorm_thresh
        """
        logger = self.logger
        logger.info('Remapping BedMachine to NE grid')

        args = [
            'ncremap',
            '-m', mapping_filename,
            '--vrb=1',
            f'--renormalize={renorm_thresh}',
            '-R', '--rgr lat_nm=y --rgr lon_nm=x',
            bedmachine_filename,
            bedmachine_on_ne_filename,
        ]
        logger.info(f'    Running: {" ".join(args)}')
        subprocess.check_call(args)

        logger.info('  Done.')

    def _combine(self):
        """
        """
        logger = self.logger
        logger.info('Combine BedMachineAntarctica and GEBCO')

        combined = xr.Dataset()
        for lat_tile in range(lat_tiles):
            for lon_tile in range(lon_tiles):
                gebco_filename = f'{gebco_prefix}_{lon_tile}_{lat_tile}.nc'
                gebco = xr.open_dataset(gebco_filename)
                tile = gebco.elevation
                tile = tile.where(tile.notnull(), 0.)
                if 'bathymetry' in combined:
                    combined['bathymetry'] = combined.bathymetry + tile
                else:
                    combined['bathymetry'] = tile

        bedmachine = xr.open_dataset(bedmachine_filename)

        alpha = (combined.lat - latmin) / (latmax - latmin)
        alpha = np.maximum(np.minimum(alpha, 1.0), 0.0)

        bedmachine_bathy = bedmachine.bathymetry
        valid = bedmachine_bathy.notnull()
        bedmachine_bathy = bedmachine_bathy.where(valid, 0.)

        combined['bathymetry'] = (
            alpha * combined.bathymetry + (1.0 - alpha) * bedmachine_bathy
        )

        mask = combined.bathymetry < 0.
        combined['bathymetry'] = combined.bathymetry.where(mask, 0.)

        for field in ['ice_draft', 'thickness']:
            combined[field] = bedmachine[field].astype(float)
        for field in ['bathymetry', 'ice_draft', 'thickness']:
            combined[field].attrs['unit'] = 'meters'

        fill = {
            'ice_mask': 0.,
            'grounded_mask': 0.,
            'ocean_mask': combined.bathymetry < 0.
        }

        for field, fill_val in fill.items():
            valid = bedmachine[field].notnull()
            combined[field] = bedmachine[field].where(valid, fill_val)

        combined['water_column'] = combined.ice_draft - combined.bathymetry
        combined.water_column.attrs['units'] = 'meters'
        combined = combined.drop_vars(['x', 'y'])

        combined.to_netcdf(combined_filename)

        logger.info('  Done.')

def _get_combined_filename(
        target_grid, datestamp, resolution,
        antarctic_filename, global_filename,
):
    """
    """

    # Parse resolution
    if target_grid == 'cubed_sphere':
        resolution_str = f'ne{resolution}'
    elif target_grid == 'lat_lon':
        resolution_str = f'{resolution}_degree'
    else:
        raise ValueError('Unknown target grid ' + target_grid)

    # Build combined filename
    combined_filename = '_'.join([
        antarctic_filename.strip('.nc'), global_filename.strip('.nc'),
        resolution_str, datestamp.strftime('%Y%m%d.nc'),
    ])

    return combined_filename

def _get_tile_filenames(global_filename, resolution, method, lat_tile, lon_tile, tiledir='tiles'):
    """
    """

    # Tiles
    global_name = global_filename.strip('.nc')
    src_filename = os.path.join(tiledir, f'{global_name}_tile_{lon_tile}_{lat_tile}.nc')
    tgt_filename = os.path.join(tiledir, f'{global_name}_on_ne_tile_{lon_tile}_{lat_tile}.nc')
    map_filename = os.path.join(tiledir, f'map_{global_name}_to_ne{resolution}_{method}_tile_{lon_tile}_{lat_tile}.nc')

    return src_filename, tgt_filename, map_filename
