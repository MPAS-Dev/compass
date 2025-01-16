import os
import pathlib

import numpy as np
import xarray as xr
from mpas_tools.cime.constants import constants
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from pyremap import MpasCellMeshDescriptor

from compass.io import symlink
from compass.parallel import run_command
from compass.step import Step


class RemapTopography(Step):
    """
    A step for remapping bathymetry and ice-shelf topography from a
    latitude-longitude grid to a global MPAS-Ocean mesh

    Attributes
    ----------
    base_mesh_step : compass.mesh.spherical.SphericalBaseStep
        The base mesh step containing input files to this step

    mesh_name : str
        The name of the MPAS mesh to include in the mapping file

    smoothing : bool, optional
        Whether smoothing will be applied as part of the remapping

    unsmoothed_topo : compass.ocean.mesh.remap_topography.RemapTopography
        A step with unsmoothed topography
    """

    def __init__(
        self, test_case, base_mesh_step, name='remap_topography', subdir=None,
        mesh_name='MPAS_mesh', smoothing=False, unsmoothed_topo=None
    ):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case this step belongs to

        base_mesh_step : compass.mesh.spherical.SphericalBaseStep
            The base mesh step containing input files to this step

        name : str, optional
            the name of the step

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        mesh_name : str, optional
            The name of the MPAS mesh to include in the mapping file

        smoothing : bool, optional
            Whether smoothing will be applied as part of the remapping

        unsmoothed_topo : compass.ocean.mesh.remap_topography.RemapTopography, optional
            A step with unsmoothed topography
        """  # noqa: E501
        super().__init__(test_case, name=name, subdir=subdir,
                         ntasks=None, min_tasks=None)
        self.base_mesh_step = base_mesh_step
        self.mesh_name = mesh_name
        self.smoothing = smoothing
        self.unsmoothed_topo = unsmoothed_topo

        self.add_output_file(filename='topography_remapped.nc')

    def setup(self):
        """
        Set up the step in the work directory, including downloading any
        dependencies.
        """
        super().setup()
        config = self.config
        section = config['remap_topography']
        topo_filename = section.get('topo_filename')
        src_scrip_filename = section.get('src_scrip_filename')

        self.add_input_file(
            filename='topography.nc',
            target=topo_filename,
            database='bathymetry_database',
        )
        self.add_input_file(
            filename='source.scrip.nc',
            target=src_scrip_filename,
            database='bathymetry_database',
        )

        base_path = self.base_mesh_step.path
        base_filename = self.base_mesh_step.config.get(
            'spherical_mesh', 'mpas_mesh_filename',
        )
        target = os.path.join(base_path, base_filename)
        self.add_input_file(filename='base_mesh.nc', work_dir_target=target)

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
        self.ntasks = config.getint('remap_topography', 'ntasks')
        self.min_tasks = config.getint('remap_topography', 'min_tasks')
        super().constrain_resources(available_resources)

    def run(self):
        """
        Run this step of the test case
        """
        super().run()
        if self._symlink_unsmoothed():
            # we symlinked to the unsmoothed topography and we're done!
            return

        config = self.config
        weight_generator = config.get('remap_topography', 'weight_generator')

        self._create_target_scrip_file()
        if weight_generator == 'tempest':
            self._partition_scrip_file('source.scrip.nc')
            self._partition_scrip_file('target.scrip.nc')
            self._create_weights_tempest()
        elif weight_generator == 'esmf':
            self._create_weights_esmf()
        else:
            msg = f'Unsupported weight generator function {weight_generator}'
            raise ValueError(msg)
        self._remap_to_target()
        self._modify_remapped_bathymetry()

    def _symlink_unsmoothed(self):
        """
        If we are smoothing but no smoothing was actually requested, symlink
        to the unsmoothed topography
        """
        if not self.smoothing or self.unsmoothed_topo is None:
            # there's no unsmoothed topogrpahy yet
            return False

        config = self.config
        section = config['remap_topography']
        expand_distance = section.getfloat('expand_distance')
        expand_factor = section.getfloat('expand_factor')

        if expand_distance != 0. or expand_factor != 1.:
            # we're doing some smoothing!
            return False

        # we already have unsmoothed topography and we're not doing
        # smoothing so we can just symlink the unsmoothed results
        out_filename = 'topography_remapped.nc'
        unsmoothed_path = self.unsmoothed_topo.work_dir
        target = os.path.join(unsmoothed_path, out_filename)
        symlink(target, out_filename)

        return True

    def _create_target_scrip_file(self):
        """
        Create target SCRIP file from MPAS mesh file.
        """
        logger = self.logger
        logger.info('Create source SCRIP file')

        if self.smoothing:
            config = self.config
            section = config['remap_topography']
            expand_distance = section.getfloat('expand_distance')
            expand_factor = section.getfloat('expand_factor')
        else:
            expand_distance = 0.
            expand_factor = 1.

        descriptor = MpasCellMeshDescriptor(
            filename='base_mesh.nc',
            mesh_name=self.mesh_name,
        )
        descriptor.to_scrip(
            'target.scrip.nc',
            expand_dist=expand_distance,
            expand_factor=expand_factor,
        )

        logger.info('  Done.')

    def _partition_scrip_file(self, in_filename):
        """
        Partition SCRIP file for parallel mbtempest use
        """
        logger = self.logger
        logger.info('Partition SCRIP file')

        stem = pathlib.Path(in_filename).stem
        h5m_filename = f'{stem}.h5m'
        part_filename = f'{stem}.p{self.ntasks}.h5m'

        # Convert source SCRIP to mbtempest
        args = [
            'mbconvert', '-B',
            in_filename,
            h5m_filename,
        ]
        check_call(args, logger)

        # Partition source SCRIP
        args = [
            'mbpart', f'{self.ntasks}',
            '-z', 'RCB',
            h5m_filename,
            part_filename,
        ]
        check_call(args, logger)

        logger.info('  Done.')

    def _create_weights_tempest(self):
        """
        Create mapping weights file using TempestRemap
        """
        logger = self.logger
        logger.info('Create weights file')

        config = self.config
        method = config.get('remap_topography', 'method')
        if method != 'conserve':
            raise ValueError(f'Unsupported method {method} for TempestRemap')

        args = [
            'mbtempest', '--type', '5',
            '--load', f'source.scrip.p{self.ntasks}.h5m',
            '--load', f'target.scrip.p{self.ntasks}.h5m',
            '--file', f'map_source_to_target_{method}.nc',
            '--weights', '--gnomonic',
            '--boxeps', '1e-9',
        ]

        run_command(
            args, self.cpus_per_task, self.ntasks,
            self.openmp_threads, self.config, self.logger,
        )

        logger.info('  Done.')

    def _create_weights_esmf(self):
        """
        Create mapping weights file using ESMF_RegridWeightGen
        """
        logger = self.logger
        logger.info('Create weights file')

        config = self.config
        method = config.get('remap_topography', 'method')

        args = [
            'ESMF_RegridWeightGen',
            '--source', 'source.scrip.nc',
            '--destination', 'target.scrip.nc',
            '--weight', f'map_source_to_target_{method}.nc',
            '--method', method,
            '--netcdf4',
            '--ignore_unmapped',
        ]

        run_command(
            args, self.cpus_per_task, self.ntasks,
            self.openmp_threads, self.config, self.logger,
        )

        logger.info('  Done.')

    def _remap_to_target(self):
        """
        Remap combined bathymetry onto MPAS target mesh
        """
        logger = self.logger
        logger.info('Remap to target')

        config = self.config
        method = config.get('remap_topography', 'method')

        # Build command args
        args = [
            'ncremap',
            '-m', f'map_source_to_target_{method}.nc',
            '--vrb=1',
            'topography.nc', 'topography_ncremap.nc',
        ]
        check_call(args, logger)

        logger.info('  Done.')

    def _modify_remapped_bathymetry(self):
        """
        Modify remapped bathymetry
        """
        logger = self.logger
        logger.info('Modify remapped bathymetry')

        config = self.config
        section = config['remap_topography']
        renorm_threshold = section.getfloat('renorm_threshold')
        ice_density = section.getfloat('ice_density')
        ocean_density = constants['SHR_CONST_RHOSW']
        g = constants['SHR_CONST_G']

        ds_in = xr.open_dataset('topography_ncremap.nc')
        ds_in = ds_in.rename({'ncol': 'nCells'})

        ds_out = xr.Dataset()
        rename = {
            'bathymetry': 'bed_elevation',
            'thickness': 'landIceThkObserved',
            'ice_mask': 'landIceFracObserved',
            'grounded_mask': 'landIceGroundedFracObserved',
            'ocean_mask': 'oceanFracObserved',
            'bathymetry_mask': 'bathyFracObserved',
        }
        for in_var, out_var in rename.items():
            ds_out[out_var] = ds_in[in_var]

        ds_out['landIceFloatingFracObserved'] = \
            ds_out.landIceFracObserved - ds_out.landIceGroundedFracObserved

        # Make sure fractions don't exceed 1
        varNames = [
            'landIceFracObserved',
            'landIceGroundedFracObserved',
            'landIceFloatingFracObserved',
            'oceanFracObserved',
            'bathyFracObserved',
        ]
        for var in varNames:
            ds_out[var] = np.minimum(ds_out[var], 1.)

        # Renormalize elevation variables
        norm = ds_out.bathyFracObserved
        valid = norm > renorm_threshold
        for var in ['bed_elevation', 'landIceThkObserved']:
            ds_out[var] = xr.where(valid, ds_out[var] / norm, 0.)

        thickness = ds_out.landIceThkObserved
        bed = ds_out.bed_elevation
        flotation_thickness = - (ocean_density / ice_density) * bed
        # not allowed to be thicker than the flotation thickness
        thickness = np.minimum(thickness, flotation_thickness)
        ds_out['landIceThkObserved'] = thickness
        ds_out['landIcePressureObserved'] = ice_density * g * thickness

        # compute the ice draft to be consistent with the land ice pressure
        # and using E3SM's density of seawater
        ds_out['landIceDraftObserved'] = \
            - (ice_density / ocean_density) * thickness

        write_netcdf(ds_out, 'topography_remapped.nc')

        logger.info('  Done.')
