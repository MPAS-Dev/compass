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

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.utility.Utility
            The test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='combine_topo')

        self.add_step(Combine(test_case=self))


class Combine(Step):
    """
    A step for combining GEBCO 2023 with BedMachineAntarctica topography
    datasets
    """

    def __init__(self, test_case):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.utility.extrap_woa.ExtraWoa
            The test case this step belongs to
        """
        super().__init__(test_case, name='combine', ntasks=None,
                         min_tasks=None)

    def setup(self):
        """
        Set up the step in the work directory, including downloading any
        dependencies.
        """
        super().setup()

        config = self.config
        section = config['combine_topo']
        antarctic_filename = section.get('antarctic_filename')
        self.add_input_file(filename=antarctic_filename,
                            target=antarctic_filename,
                            database='bathymetry_database')
        global_filename = section.get('global_filename')
        self.add_input_file(filename=global_filename,
                            target=global_filename,
                            database='bathymetry_database')

        cobined_filename = section.get('cobined_filename')
        self.add_output_file(filename=cobined_filename)

        self.ntasks = section.getint('ntasks')
        self.min_tasks = section.getint('min_tasks')

    def run(self):
        """
        Run this step of the test case
        """
        self._downsample_gebco()
        self._modify_bedmachine()
        # we will work with bedmachine data at its original resolution
        # self._downsample_bedmachine()
        self._remap_bedmachine()
        self._combine()

    def _downsample_gebco(self):
        """
        Average GEBCO to 0.0125 degree grid. GEBCO is on 15" grid, so average
        every 3x3 block of cells
        """
        config = self.config
        logger = self.logger

        section = config['combine_topo']
        in_filename = section.get('global_filename')
        out_filename = 'GEBCO_2023_0.0125_degree.nc'

        gebco = xr.open_dataset(in_filename)

        nlon = gebco.sizes['lon']
        nlat = gebco.sizes['lat']

        block = 3
        norm = 1.0 / block**2

        nx = nlon // block
        ny = nlat // block

        chunks = 2
        nxchunk = nlon // chunks
        nychunk = nlat // chunks

        gebco = gebco.chunk({'lon': nxchunk, 'lat': nychunk})

        bathymetry = np.zeros((ny, nx))

        logger.info('Averaging GEBCO to 0.0125 degree grid')
        widgets = [progressbar.Percentage(), ' ',
                   progressbar.Bar(), ' ', progressbar.ETA()]
        bar = progressbar.ProgressBar(widgets=widgets,
                                      maxval=chunks**2).start()
        for ychunk in range(chunks):
            for xchunk in range(chunks):
                gebco_chunk = gebco.isel(
                    lon=slice(nxchunk * xchunk, nxchunk * (xchunk + 1)),
                    lat=slice(nychunk * ychunk, nychunk * (ychunk + 1)))
                elevation = gebco_chunk.elevation.values
                nxblock = nxchunk // block
                nyblock = nychunk // block
                bathy_block = np.zeros((nyblock, nxblock))
                for y in range(block):
                    for x in range(block):
                        bathy_block += elevation[y::block, x::block]
                bathy_block *= norm
                xmin = xchunk * nxblock
                xmax = (xchunk + 1) * nxblock
                ymin = ychunk * nyblock
                ymax = (ychunk + 1) * nyblock
                bathymetry[ymin:ymax, xmin:xmax] = bathy_block
                bar.update(ychunk * chunks + xchunk + 1)
        bar.finish()

        lon_corner = np.linspace(-180., 180., bathymetry.shape[1] + 1)
        lat_corner = np.linspace(-90., 90., bathymetry.shape[0] + 1)
        lon = 0.5 * (lon_corner[0:-1] + lon_corner[1:])
        lat = 0.5 * (lat_corner[0:-1] + lat_corner[1:])
        gebco_low = xr.Dataset({'bathymetry': (['lat', 'lon'], bathymetry)},
                               coords={'lon': (['lon',], lon),
                                       'lat': (['lat',], lat)})
        gebco_low.attrs = gebco.attrs
        gebco_low.lon.attrs = gebco.lon.attrs
        gebco_low.lat.attrs = gebco.lat.attrs
        gebco_low.bathymetry.attrs = gebco.elevation.attrs
        gebco_low.to_netcdf(out_filename)

    def _modify_bedmachine(self):
        """
        Modify BedMachineAntarctica to compute the fields needed by MPAS-Ocean
        """
        logger = self.logger
        logger.info('Modifying BedMachineAntarctica with MPAS-Ocean names')

        config = self.config

        section = config['combine_topo']
        in_filename = section.get('antarctic_filename')
        out_filename = 'BedMachineAntarctica-v3_mod.nc'
        bedmachine = xr.open_dataset(in_filename)
        mask = bedmachine.mask
        ice_mask = (mask != 0).astype(float)
        ocean_mask = (np.logical_or(mask == 0, mask == 3)).astype(float)
        grounded_mask = np.logical_or(np.logical_or(mask == 1, mask == 2),
                                      mask == 4).astype(float)

        bedmachine['bathymetry'] = bedmachine.bed.where(ocean_mask, 0.)
        bedmachine['ice_draft'] = \
            (bedmachine.surface -
                bedmachine.thickness).where(ocean_mask, 0.)
        bedmachine.ice_draft.attrs['units'] = 'meters'
        bedmachine['thickness'] = \
            bedmachine.thickness.where(ocean_mask, 0.)

        bedmachine['ice_mask'] = ice_mask
        bedmachine['grounded_mask'] = grounded_mask
        bedmachine['ocean_mask'] = ocean_mask

        varlist = ['bathymetry', 'ice_draft', 'thickness', 'ice_mask',
                   'grounded_mask', 'ocean_mask']

        bedmachine = bedmachine[varlist]

        bedmachine.to_netcdf(out_filename)
        logger.info('  Done.')

    def _downsample_bedmachine(self):
        """
        Downsample bedmachine from 0.5 to 1 km grid
        """
        logger = self.logger
        logger.info('Downsample BedMachineAntarctica from 500 m to 1 km')

        in_filename = 'BedMachineAntarctica-v3_mod.nc'
        out_filename = 'BedMachineAntarctica-v3_1k.nc'
        bedmachine = xr.open_dataset(in_filename)
        x = bedmachine.x.values
        y = bedmachine.y.values

        nx = len(x) // 2
        ny = len(y) // 2
        x = 0.5 * (x[0:2 * nx:2] + x[1:2 * nx:2])
        y = 0.5 * (y[0:2 * ny:2] + y[1:2 * ny:2])
        bedmachine1k = xr.Dataset()
        for field in bedmachine.data_vars:
            in_array = bedmachine[field].values
            out_array = np.zeros((ny, nx))
            for yoffset in range(2):
                for xoffset in range(2):
                    out_array += 0.25 * \
                        in_array[yoffset:2 * ny:2, xoffset:2 * nx:2]
            da = xr.DataArray(out_array, dims=('y', 'x'),
                              coords={'x': (('x',), x),
                                      'y': (('y',), y)})
            bedmachine1k[field] = da
            bedmachine1k[field].attrs = bedmachine[field].attrs

        bedmachine1k.to_netcdf(out_filename)
        logger.info('  Done.')

    def _remap_bedmachine(self):
        """
        Remap BedMachine Antarctica to GEBCO lat-lon grid
        """
        logger = self.logger
        logger.info('Remap BedMachineAntarctica to GEBCO 1/80 deg grid')

        config = self.config

        section = config['combine_topo']
        renorm_thresh = section.getfloat('renorm_thresh')

        in_filename = 'BedMachineAntarctica-v3_mod.nc'
        out_filename = 'BedMachineAntarctica_on_GEBCO_low.nc'
        gebco_filename = 'GEBCO_2023_0.0125_degree.nc'

        projection = pyproj.Proj('+proj=stere +lat_ts=-71.0 +lat_0=-90 '
                                 '+lon_0=0.0 +k_0=1.0 +x_0=0.0 +y_0=0.0 '
                                 '+ellps=WGS84')

        bedmachine = xr.open_dataset(in_filename)
        x = bedmachine.x.values
        y = bedmachine.y.values

        in_descriptor = ProjectionGridDescriptor.create(
            projection, x, y, 'BedMachineAntarctica_500m')

        out_descriptor = LatLonGridDescriptor.read(fileName=gebco_filename)

        mapping_filename = \
            'map_BedMachineAntarctica_500m_to_GEBCO_0.0125deg_bilinear.nc'

        remapper = Remapper(in_descriptor, out_descriptor, mapping_filename)
        remapper.build_mapping_file(method='bilinear', mpiTasks=self.ntasks,
                                    esmf_parallel_exec='srun', tempdir='.')
        bedmachine = xr.open_dataset(in_filename)
        bedmachine_on_gebco_low = remapper.remap(bedmachine)

        for field in ['bathymetry', 'ice_draft', 'thickness']:
            bedmachine_on_gebco_low[field].attrs['unit'] = 'meters'

        # renormalize the fields based on the ocean masks
        ocean_mask = bedmachine_on_gebco_low.ocean_mask

        valid = ocean_mask > renorm_thresh
        norm = ocean_mask.where(valid, 1.)
        norm = 1. / norm
        for field in ['bathymetry', 'ice_draft', 'thickness']:
            bedmachine_on_gebco_low[field] = \
                norm * bedmachine_on_gebco_low[field].where(valid, 0.)

        bedmachine_on_gebco_low.to_netcdf(out_filename)
        logger.info('  Done.')

    def _combine(self):
        """
        Combine GEBCO with BedMachine Antarctica
        """
        logger = self.logger
        logger.info('Combine BedMachineAntarctica and GEBCO')

        config = self.config
        section = config['combine_topo']

        latmin = section.getfloat('latmin')
        latmax = section.getfloat('latmax')
        cobined_filename = section.get('cobined_filename')

        gebco_filename = 'GEBCO_2023_0.0125_degree.nc'
        gebco = xr.open_dataset(gebco_filename)

        bedmachine_filename = 'BedMachineAntarctica_on_GEBCO_low.nc'
        bedmachine = xr.open_dataset(bedmachine_filename)

        combined = xr.Dataset()
        alpha = (gebco.lat - latmin) / (latmax - latmin)
        alpha = np.maximum(np.minimum(alpha, 1.0), 0.0)

        bedmachine_bathy = bedmachine.bathymetry
        valid = bedmachine_bathy.notnull()
        bedmachine_bathy = bedmachine_bathy.where(valid, 0.)

        combined['bathymetry'] = \
            alpha * gebco.bathymetry.where(gebco.bathymetry < 0, 0.) + \
            (1.0 - alpha) * bedmachine_bathy
        for field in ['ice_draft', 'thickness']:
            combined[field] = bedmachine[field]
        for field in ['bathymetry', 'ice_draft', 'thickness']:
            combined[field].attrs['unit'] = 'meters'

        fill = {'ice_mask': 0., 'grounded_mask': 0.,
                'ocean_mask': combined['bathymetry'] < 0.}

        for field, fill_val in fill.items():
            valid = bedmachine[field].notnull()
            combined[field] = bedmachine[field].where(valid, fill_val)

        combined['water_column'] = \
            combined['ice_draft'] - combined['bathymetry']
        combined.water_column.attrs['units'] = 'meters'

        combined.to_netcdf(cobined_filename)
        logger.info('  Done.')

        diff = xr.Dataset()

        diff['bathymetry'] = \
            gebco.bathymetry.where(gebco.bathymetry < 0, 0.) - \
            bedmachine.bathymetry
        diff.to_netcdf('gebco_minus_bedmachine.nc')
