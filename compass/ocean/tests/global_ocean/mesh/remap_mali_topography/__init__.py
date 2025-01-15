import os

import numpy as np
import xarray as xr
from mpas_tools.cime.constants import constants
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from pyremap import MpasCellMeshDescriptor

from compass.ocean.mesh.remap_topography import RemapTopography
from compass.parallel import run_command


class RemapMaliTopography(RemapTopography):
    """
    A step for remapping bathymetry and ice-shelf topography from a MALI
    datase to a global MPAS-Ocean mesh and combining it with a global
    topography dataset

    Attributes
    ----------
    mali_ais_topo : str
        Short name for the MALI dataset to use for Antarctic Ice Sheet
        topography

    ocean_includes_grounded : bool, optional
        Whether to include grounded cells that are below sea level in the
        ocean domain
    """

    def __init__(self, test_case, base_mesh_step, mesh_name, mali_ais_topo,
                 ocean_includes_grounded):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case this step belongs to

        base_mesh_step : compass.mesh.spherical.SphericalBaseStep
            The base mesh step containing input files to this step

        remap_topography : compass.ocean.mesh.remap_topography.RemapTopography
            A step for remapping topography. If provided, the remapped
            topography is used to determine the land mask

        mesh_name : str
            The name of the MPAS mesh to include in the mapping file

        mali_ais_topo : str, optional
            Short name for the MALI dataset to use for Antarctic Ice Sheet
            topography

        ocean_includes_grounded : bool
            Whether to include grounded cells that are below sea level in the
            ocean domain
        """
        super().__init__(test_case=test_case, base_mesh_step=base_mesh_step,
                         mesh_name=mesh_name)
        self.mali_ais_topo = mali_ais_topo
        self.ocean_includes_grounded = ocean_includes_grounded

        self.add_output_file(filename='mali_topography_remapped.nc')

    def setup(self):
        """
        Set up the step in the work directory, including downloading any
        dependencies.
        """
        super().setup()

        mali_filename = self.config.get('remap_mali_topography',
                                        'mali_filename')
        self.add_input_file(
            filename='mali_topography_orig.nc',
            target=mali_filename,
            database='mali_topo')

    def run(self):
        """
        Run this step of the test case
        """
        super().run()
        self._remap_mali_topo()
        self._combine_topo()

    def _remap_mali_topo(self):
        in_mesh_name = self.mali_ais_topo
        in_descriptor = MpasCellMeshDescriptor(
            fileName='mali_topography_orig.nc',
            meshName=in_mesh_name)
        in_descriptor.format = 'NETCDF3_64BIT'
        in_descriptor.to_scrip('mali.scrip.nc')
        self._partition_scrip_file('mali.scrip.nc')

        out_mesh_name = self.mesh_name
        out_descriptor = MpasCellMeshDescriptor(
            fileName='base_mesh.nc',
            meshName=out_mesh_name)
        out_descriptor.format = 'NETCDF3_64BIT'
        out_descriptor.to_scrip('mpaso.scrip.nc')
        self._partition_scrip_file('mpaso.scrip.nc')

        map_filename = \
            f'map_{in_mesh_name}_to_{out_mesh_name}_mbtraave.nc'

        self._create_mali_to_mpaso_weights(map_filename)

        self._modify_mali_topo()

        self._remap_mali_to_mpaso(map_filename)

    def _modify_mali_topo(self):
        """
        Modify MALI topography to have desired fields and names
        """
        logger = self.logger
        config = self.config
        logger.info('Modifying MALI topography fields and names')

        ds_mali = xr.open_dataset('mali_topography_orig.nc')
        if 'Time' in ds_mali.dims:
            ds_mali = ds_mali.isel(Time=0)
        bed = ds_mali.bedTopography
        thickness = ds_mali.thickness

        ice_density = config.getfloat('remap_topography', 'ice_density')
        ocean_density = config.getfloat('remap_topography', 'ocean_density')
        sea_level = config.getfloat('remap_topography', 'sea_level')

        g = constants['SHR_CONST_G']

        draft = - (ice_density / ocean_density) * thickness

        ice_mask = ds_mali.thickness > 0
        floating_mask = np.logical_and(
            ice_mask,
            ice_density / ocean_density * thickness <= sea_level - bed)
        grounded_mask = np.logical_and(ice_mask, np.logical_not(floating_mask))

        if self.ocean_includes_grounded:
            ocean_mask = bed < sea_level
        else:
            ocean_mask = np.logical_and(np.logical_not(grounded_mask),
                                        bed < sea_level)

        lithop = ice_density * g * thickness
        ice_frac = xr.where(ice_mask, 1., 0.)
        grounded_frac = xr.where(grounded_mask, 1., 0.)
        ocean_frac = xr.where(ocean_mask, 1., 0.)
        mali_frac = xr.DataArray(data=np.ones(ds_mali.sizes['nCells']),
                                 dims='nCells')

        # we will remap conservatively
        ds_in = xr.Dataset()
        ds_in['bed_elevation'] = bed
        ds_in['landIceThkObserved'] = thickness
        ds_in['landIcePressureObserved'] = lithop
        ds_in['landIceDraftObserved'] = draft
        ds_in['landIceFrac'] = ice_frac
        ds_in['landIceGroundedFrac'] = grounded_frac
        ds_in['oceanFrac'] = ocean_frac
        ds_in['maliFrac'] = mali_frac

        write_netcdf(ds_in, 'mali_topography_mod.nc')

        logger.info('  Done.')

    def _create_mali_to_mpaso_weights(self, map_filename):
        """
        Create mapping weights file using TempestRemap
        """
        logger = self.logger
        logger.info('Create weights file')

        args = [
            'mbtempest', '--type', '5',
            '--load', f'mali.scrip.p{self.ntasks}.h5m',
            '--load', f'mpaso.scrip.p{self.ntasks}.h5m',
            '--file', map_filename,
            '--weights', '--gnomonic',
            '--boxeps', '1e-9',
        ]

        run_command(
            args, self.cpus_per_task, self.ntasks,
            self.openmp_threads, self.config, self.logger,
        )

        logger.info('  Done.')

    def _remap_mali_to_mpaso(self, map_filename):
        """
        Remap combined bathymetry onto MPAS target mesh
        """
        logger = self.logger
        logger.info('Remap to target')

        args = [
            'ncremap',
            '-m', map_filename,
            '--vrb=1',
            'mali_topography_mod.nc', 'mali_topography_ncremap.nc',
        ]
        check_call(args, logger)

        ds_remapped = xr.open_dataset('mali_topography_ncremap.nc')
        ds_out = xr.Dataset()
        for var_name in ds_remapped:
            var = ds_remapped[var_name]
            if 'Frac' in var:
                # copy the fractional variable, making sure it doesn't
                # exceed 1
                var = np.minimum(var, 1.)
            ds_out[var_name] = var

        write_netcdf(ds_out, 'mali_topography_remapped.nc')

        logger.info('  Done.')

    def _combine_topo(self):
        os.rename('topography_remapped.nc',
                  'bedmachine_topography_remapped.nc')
        ds_mali = xr.open_dataset('mali_topography_remapped.nc')
        ds_mali = ds_mali.reset_coords(['lat', 'lon'], drop=True)
        ds_bedmachine = xr.open_dataset('bedmachine_topography_remapped.nc')
        ds_bedmachine = ds_bedmachine.reset_coords(['lat', 'lon'], drop=True)

        ds_out = xr.Dataset(ds_bedmachine)
        # for now, just use the land-ice pressure, draft and thickness from
        # MALI
        for var in ['landIcePressureObserved', 'landIceDraftObserved',
                    'landIceThkObserved']:
            ds_out[var] = ds_mali[var]

        ds_out['landIceFracObserved'] = ds_mali['landIceFrac']

        ds_out['landIceFloatingFracObserved'] = (
            ds_mali['landIceFrac'] -
            ds_mali['landIceGroundedFrac'])

        for var in ['maliFrac']:
            mali_field = ds_mali[var]
            mali_field = xr.where(mali_field.notnull(), mali_field, 0.)
            ds_out[var] = mali_field

        # for now, blend topography at calving fronts, but we may want a
        # smoother blend in the future
        alpha = ds_mali.landIceFrac
        ds_out['bed_elevation'] = (
            alpha * ds_mali.bed_elevation +
            (1.0 - alpha) * ds_bedmachine.bed_elevation)

        alpha = ds_out.maliFrac
        ds_out['oceanFracObserved'] = (
            alpha * ds_mali.oceanFrac +
            (1.0 - alpha) * ds_bedmachine.oceanFracObserved)

        ds_out['ssh'] = ds_out.landIceDraftObserved

        write_netcdf(ds_out, 'topography_remapped.nc')
