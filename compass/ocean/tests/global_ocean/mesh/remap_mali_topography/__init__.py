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
    """

    def __init__(self, test_case, base_mesh_step, mesh_name, mali_ais_topo):
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
        """
        super().__init__(test_case=test_case, base_mesh_step=base_mesh_step,
                         mesh_name=mesh_name)
        self.mali_ais_topo = mali_ais_topo

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
            filename='mali_topography.nc',
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
        config = self.config
        logger = self.logger

        in_mesh_name = self.mali_ais_topo
        in_descriptor = MpasCellMeshDescriptor(fileName='mali_topography.nc',
                                               meshName=in_mesh_name)
        in_descriptor.format = 'NETCDF3_64BIT'
        in_descriptor.to_scrip('mali_scrip.nc')

        out_mesh_name = self.mesh_name
        out_descriptor = MpasCellMeshDescriptor(fileName='base_mesh.nc',
                                                meshName=out_mesh_name)
        out_descriptor.format = 'NETCDF3_64BIT'
        out_descriptor.to_scrip('mpaso_scrip.nc')

        args = ['mbtempest',
                '--type', '5',
                '--load', 'mali_scrip.nc',
                '--load', 'mpaso_scrip.nc',
                '--intx', 'moab_intx_mali_mpaso.h5m',
                '--rrmgrids']

        run_command(args=args, cpus_per_task=self.cpus_per_task,
                    ntasks=self.ntasks, openmp_threads=self.openmp_threads,
                    config=config, logger=logger)

        ds_mali = xr.open_dataset('mali_topography.nc')
        if 'Time' in ds_mali.dims:
            ds_mali = ds_mali.isel(Time=0)
        bed = ds_mali.bedTopography
        thickness = ds_mali.thickness

        ice_density = config.getfloat('remap_topography', 'ice_density')

        mali_ice_density = ds_mali.attrs['config_ice_density']
        mali_ocean_density = ds_mali.attrs['config_ocean_density']
        sea_level = ds_mali.attrs['config_sea_level']

        g = constants['SHR_CONST_G']
        ocean_density = constants['SHR_CONST_RHOSW']

        if ice_density != mali_ice_density:
            raise ValueError('Ice density from the config option in '
                             '[remap_topography] does not match the value '
                             'from MALI config_ice_density')
        if ocean_density != mali_ocean_density:
            logger.warn('\nWARNING: Ocean density from SHR_CONST_RHOSW does '
                        'not match the value from MALI config_ocean_density\n')

        draft = - (ice_density / ocean_density) * thickness

        ice_mask = ds_mali.thickness > 0
        floating_mask = np.logical_and(
            ice_mask,
            ice_density / mali_ocean_density * thickness <= sea_level - bed)
        grounded_mask = np.logical_and(ice_mask, np.logical_not(floating_mask))
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

        mbtempest_args = {'conserve': ['--order', '1',
                                       '--order', '1',
                                       '--fvmethod', 'none',
                                       '--rrmgrids'],
                          'bilinear': ['--order', '1',
                                       '--order', '1',
                                       '--fvmethod', 'bilin']}

        suffix = {'conserve': 'mbtraave',
                  'bilinear': 'mbtrbilin'}

        method = 'conserve'
        mapping_file_name = \
            f'map_{in_mesh_name}_to_{out_mesh_name}_{suffix[method]}.nc'

        # split the parallel executable into constituents in case it
        # includes flags
        args = ['mbtempest',
                '--type', '5',
                '--load', 'mali_scrip.nc',
                '--load', 'mpaso_scrip.nc',
                '--intx', 'moab_intx_mali_mpaso.h5m',
                '--weights',
                '--method', 'fv',
                '--method', 'fv',
                '--file', mapping_file_name] + mbtempest_args[method]

        if method == 'bilinear':
            # unhappy in parallel for now
            check_call(args, logger)
        else:

            run_command(args=args, cpus_per_task=self.cpus_per_task,
                        ntasks=self.ntasks,
                        openmp_threads=self.openmp_threads,
                        config=config, logger=logger)

        in_filename = f'mali_topography_{method}.nc'
        out_filename = f'mali_topography_ncremap_{method}.nc'
        write_netcdf(ds_in, in_filename)

        # remapping with the -P mpas flag leads to undesired
        # renormalization
        args = ['ncremap',
                '-m', mapping_file_name,
                '--vrb=1',
                in_filename,
                out_filename]
        check_call(args, logger)

        ds_remapped = xr.open_dataset(out_filename)
        ds_out = xr.Dataset()
        for var_name in ds_remapped:
            var = ds_remapped[var_name]
            if 'Frac' in var:
                # copy the fractional variable, making sure it doesn't
                # exceed 1
                var = np.minimum(var, 1.)
            ds_out[var_name] = var

        write_netcdf(ds_out, 'mali_topography_remapped.nc')

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
