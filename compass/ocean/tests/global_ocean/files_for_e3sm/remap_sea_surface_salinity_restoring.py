import os
import pathlib

import numpy as np
import xarray as xr
from mpas_tools.logging import check_call
from pyremap import MpasCellMeshDescriptor

from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm.files_for_e3sm_step import (  # noqa: E501
    FilesForE3SMStep,
)
from compass.parallel import run_command


class RemapSeaSurfaceSalinityRestoring(FilesForE3SMStep):
    """
    A step for for remapping sea surface salinity (SSS) from WOA23 to the
    current MPAS mesh
    """
    def __init__(self, test_case):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case,
                         name='remap_sea_surface_salinity_restoring')

        self.add_input_file(
            target='woa23_decav_ne300_sss_monthly_extrap.20250114.nc',
            database='initial_condition_database')

        self.add_input_file(
            target='ne300_20250114.scrip.nc',
            database='initial_condition_database')

        self.add_output_file(filename='sss.WOA23_monthlyClimatology.nc')

    def setup(self):
        """
        Set resources from config options
        """
        super().setup()
        section = self.config['files_for_e3sm']
        self.ntasks = section.getint('remap_sss_ntasks')
        self.min_tasks = section.getint('remap_sss_min_tasks')

    def constrain_resources(self, available_resources):
        """
        Constrain ``cpus_per_task`` and ``ntasks`` based on the number of
        cores available to this step

        Parameters
        ----------
        available_resources : dict
            The total number of cores available to the step
        """
        section = self.config['files_for_e3sm']
        self.ntasks = section.getint('remap_sss_ntasks')
        self.min_tasks = section.getint('remap_sss_min_tasks')
        super().constrain_resources(available_resources)

    def run(self):
        """
        Run this step of the test case
        """
        super().run()

        in_filename = self.inputs[0]
        src_scrip_filename = self.inputs[1]

        prefix = 'sss.WOA23_monthlyClimatology'
        suffix = f'{self.mesh_short_name}.{self.creation_date}'

        out_filename = f'{prefix}.nc'
        dest_filename = f'{prefix}.{suffix}.nc'

        mesh_filename = 'restart.nc'
        mesh_name = self.mesh_short_name

        target_scrip_filename = self._create_target_scrip_file(
            mesh_filename, mesh_name)

        mapping_filename = \
            f'map_ne300_to_{mesh_name}_mbtraave.nc'

        stem = pathlib.Path(out_filename).stem
        remap_filename = f'{stem}_after_remap.nc'

        src_partition_filename = self._partition_scrip_file(
            src_scrip_filename)
        target_partition_filename = self._partition_scrip_file(
            target_scrip_filename)
        self._create_weights_tempest(src_partition_filename,
                                     target_partition_filename,
                                     mapping_filename)
        self._remap_to_target(in_filename, remap_filename, mapping_filename)

        self._modify_remapped_sss(remap_filename, out_filename)

        symlink(
            os.path.abspath(out_filename),
            f'{self.ocean_inputdata_dir}/{dest_filename}')

    def _create_target_scrip_file(self, target_mesh_filename, mesh_name):
        """
        Create target SCRIP file from MPAS mesh file
        """
        logger = self.logger
        logger.info('Create target SCRIP file')

        config = self.config
        section = config['files_for_e3sm']
        min_lat = np.deg2rad(section.getfloat('sss_smoothing_min_lat'))
        max_dist = section.getfloat('sss_smoothing_max_dist')

        scrip_filename = f'{mesh_name}.scrip.nc'

        ds_mesh = xr.open_dataset(target_mesh_filename)
        lat_cell = ds_mesh.latCell

        expand_dist = xr.zeros_like(lat_cell)
        mask = lat_cell >= min_lat
        # goes from 1 at the North pole to zero at min_lat
        alpha = (lat_cell - min_lat) / (0.5 * np.pi - min_lat)
        expand_dist[mask] = alpha[mask] * max_dist

        ds_out = xr.Dataset()
        ds_out['expandDist'] = expand_dist
        self.write_netcdf(ds_out, 'expandDist.nc')

        descriptor = MpasCellMeshDescriptor(
            filename=target_mesh_filename,
            mesh_name=mesh_name,
        )
        descriptor.to_scrip(scrip_filename, expand_dist=expand_dist)

        logger.info('  Done.')
        return scrip_filename

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
        # run in "parallel" with one task and one thread for Intel-MPI support
        run_command(args, 1, 1, 1, self.config, logger)

        # Partition source SCRIP
        args = [
            'mbpart', f'{self.ntasks}',
            '-z', 'RCB',
            h5m_filename,
            part_filename,
        ]
        # run in "parallel" with one task and one thread for Intel-MPI support
        run_command(args, 1, 1, 1, self.config, logger)

        logger.info('  Done.')
        return part_filename

    def _create_weights_tempest(self, src_partition_filename,
                                target_partition_filename,
                                mapping_filename):
        """
        Create mapping weights file using TempestRemap
        """
        logger = self.logger
        logger.info('Create weights file')

        args = [
            'mbtempest', '--type', '5',
            '--load', src_partition_filename,
            '--load', target_partition_filename,
            '--file', mapping_filename,
            '--weights', '--gnomonic',
            '--boxeps', '1e-9',
        ]

        run_command(
            args, self.min_cpus_per_task, self.ntasks,
            self.openmp_threads, self.config, self.logger
        )

        logger.info('  Done.')

    def _remap_to_target(self, in_filename, remap_filename, mapping_filename):
        """
        Remap SSS onto MPAS target mesh
        """
        logger = self.logger
        logger.info('Remap to target')

        # Build command args
        args = [
            'ncremap',
            '-m', mapping_filename,
            '--vrb=1',
            in_filename, remap_filename,
        ]
        check_call(args, logger)

        logger.info('  Done.')

    def _modify_remapped_sss(self, remap_filename, out_filename):
        """
        Modify remapped SSS
        """
        logger = self.logger
        ds = xr.open_dataset(remap_filename)
        logger.info('Removing lat/lon bounds variables...')
        drop = [var for var in ds if 'nv' in ds[var].dims]
        ds = ds.drop_vars(drop)
        logger.info('Renaming dimensions and variables...')
        rename = dict(SALT='surfaceSalinityMonthlyClimatologyValue')
        ds = ds.rename(rename)
        self.write_netcdf(ds, out_filename)
