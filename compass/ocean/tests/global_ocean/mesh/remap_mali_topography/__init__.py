import os

import numpy as np
import xarray as xr
from mpas_tools.cime.constants import constants
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from pyremap import MpasCellMeshDescriptor

from compass.io import symlink
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

    def __init__(
            self,
            test_case,
            base_mesh_step,
            mesh_name,
            mali_ais_topo,
            ocean_includes_grounded,
            name,
            smoothing,
            unsmoothed_topo=None):
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

        mali_ais_topo : str
            Short name for the MALI dataset to use for Antarctic Ice Sheet
            topography

        ocean_includes_grounded : bool
            Whether to include grounded cells that are below sea level in the
            ocean domain

        name : str, optional
            the name of the step

        smoothing : bool, optional
            Whether smoothing will be applied as part of the remapping

        unsmoothed_topo : compass.ocean.mesh.remap_topography.RemapTopography, optional
            A step with unsmoothed topography
        """  # noqa: E501
        super().__init__(
            test_case=test_case,
            base_mesh_step=base_mesh_step,
            mesh_name=mesh_name,
            name=name,
            smoothing=smoothing,
            unsmoothed_topo=unsmoothed_topo,
        )
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

        if not self.config.has_option('remap_topography', 'ocean_density'):
            raise ValueError(
                'You must be using a mesh that defines [remap_topography]/'
                'ocean_density in the config file')

    def run(self):
        """
        Run this step of the test case
        """
        super().run()
        if self.symlinked_to_unsmoothed:
            # we already have unsmoothed topography and we're not doing
            # smoothing so we can just symlink the unsmoothed results
            out_filename = 'mali_topography_remapped.nc'
            unsmoothed_path = self.unsmoothed_topo.work_dir
            target = os.path.join(unsmoothed_path, out_filename)
            symlink(target, out_filename)
        else:
            self._remap_mali_topo()
            self._combine_topo()

    def _remap_mali_topo(self):
        in_mesh_name = self.mali_ais_topo
        in_descriptor = MpasCellMeshDescriptor(
            filename='mali_topography_orig.nc',
            mesh_name=in_mesh_name)
        in_descriptor.format = 'NETCDF3_64BIT'
        in_descriptor.to_scrip('mali.scrip.nc')
        self._partition_scrip_file('mali.scrip.nc')

        out_mesh_name = self.mesh_name

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

        draft = - (ice_density / ocean_density) * thickness + sea_level

        ice_mask = ds_mali.thickness > 0
        floating_mask = np.logical_and(ice_mask, draft > bed)
        grounded_mask = np.logical_and(ice_mask, np.logical_not(floating_mask))

        # draft is determined by the bed where the ice is grounded and by
        # flotation where the ice is not grounded (floating or no ice)
        draft = xr.where(grounded_mask, bed, draft)

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
        ds_in['bed_elevation'] = ocean_frac * bed
        ds_in['landIceThkObserved'] = ocean_frac * thickness
        ds_in['landIcePressureObserved'] = ocean_frac * lithop
        ds_in['landIceDraftObserved'] = ocean_frac * draft
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

        # target h5m file was created in parent class
        args = [
            'mbtempest', '--type', '5',
            '--load', f'mali.scrip.p{self.ntasks}.h5m',
            '--load', f'target.scrip.p{self.ntasks}.h5m',
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
        config = self.config
        section = config['remap_topography']
        renorm_threshold = section.getfloat('renorm_threshold')

        args = [
            'ncremap',
            '-m', map_filename,
            '--vrb=1',
            'mali_topography_mod.nc', 'mali_topography_ncremap.nc',
        ]
        check_call(args, logger)

        ds_remapped = xr.open_dataset('mali_topography_ncremap.nc')
        ds_out = xr.Dataset()
        for var_name in ds_remapped.keys():
            var = ds_remapped[var_name]
            if 'Frac' in var_name:
                # copy the fractional variable, making sure it doesn't
                # exceed 1
                var = np.minimum(var, 1.)
            ds_out[var_name] = var

        # renormalize topography variables based on ocean fraction
        ocean_frac = ds_out['oceanFrac']
        ocean_mask = ocean_frac > renorm_threshold
        norm = xr.where(ocean_mask, 1.0 / ocean_frac, 0.0)
        for var_name in [
            'bed_elevation',
            'landIceThkObserved',
            'landIcePressureObserved',
            'landIceDraftObserved'
        ]:
            attrs = ds_out[var_name].attrs
            ds_out[var_name] = ds_out[var_name] * norm
            ds_out[var_name].attrs = attrs

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
            ds_mali['landIceGroundedFrac']
        )

        ds_out['landIceGroundedFracObserved'] = ds_mali['landIceGroundedFrac']

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

        # our topography blending technique can lead to locations where the
        # ice draft is slightly below the bed elevation; we correct that here
        ds_out['landIceDraftObserved'] = xr.where(
            ds_out['landIceDraftObserved'] < ds_out['bed_elevation'],
            ds_out['bed_elevation'],
            ds_out['landIceDraftObserved'])

        alpha = ds_out.maliFrac
        # NOTE: MALI's ocean fraction is already scaled by the MALI fraction
        ds_out['oceanFracObserved'] = xr.where(
            ds_mali.bed_elevation > 0.,
            0.,
            ds_mali.oceanFrac +
            (1.0 - alpha) * ds_bedmachine.oceanFracObserved)

        # limit land ice floatation fraction to ocean fraction -- because of
        # machine-precision noise in the combined ocean fraciton,
        # land ice floating fraction can exceed ocean fraction by ~1e-15
        ds_out['landIceFloatingFracObserved'] = np.minimum(
            ds_out['landIceFloatingFracObserved'],
            ds_out['oceanFracObserved']
        )

        ds_out['ssh'] = ds_out.landIceDraftObserved

        write_netcdf(ds_out, 'topography_remapped.nc')
