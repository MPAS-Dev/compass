from os.path import exists
from shutil import copyfile

import netCDF4
import numpy as np
from mpas_tools.logging import check_call
from mpas_tools.scrip.from_mpas import scrip_from_mpas

from compass.landice.mesh import (
    build_cell_width,
    build_mali_mesh,
    make_region_masks,
)
from compass.model import make_graph_file
from compass.step import Step


class Mesh(Step):
    """
    A step for creating a mesh and initial condition for greenland test cases

    Attributes
    ----------
    mesh_type : str
        The resolution or mesh type of the test case
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        mesh_type : str
            The resolution or mesh type of the test case
        """
        super().__init__(test_case=test_case, name='mesh', cpus_per_task=128,
                         min_cpus_per_task=1)

        self.add_output_file(filename='graph.info')
        self.add_output_file(filename='GIS.nc')
        self.add_input_file(
            filename='greenland_1km_2024_01_29.epsg3413.icesheetonly.nc',
            target='greenland_1km_2024_01_29.epsg3413.icesheetonly.nc',
            database='')
        self.add_input_file(filename='greenland_2km_2024_01_29.epsg3413.nc',
                            target='greenland_2km_2024_01_29.epsg3413.nc',
                            database='')

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        mesh_name = 'GIS.nc'
        section_name = 'mesh'
        config = self.config
        section = config[section_name]
        data_path = section.get('data_path')
        nProcs = section.get('nProcs')

        logger.info('calling build_cell_width')
        cell_width, x1, y1, geom_points, geom_edges, floodMask = \
            build_cell_width(
                self, section_name=section_name,
                gridded_dataset='greenland_2km_2024_01_29.epsg3413.nc',
                flood_fill_start=[100, 700])

        build_mali_mesh(
            self, cell_width, x1, y1, geom_points, geom_edges,
            mesh_name=mesh_name, section_name=section_name,
            gridded_dataset='greenland_1km_2024_01_29.epsg3413.icesheetonly.nc',  # noqa
            projection='gis-gimp', geojson_file=None)

        # Create scrip files if they don't already exist
        if exists(data_path + '/BedMachineGreenland-v5.scrip.nc'):
            logger.info('BedMachine script file exists;'
                        ' skipping file creation')
        else:
            logger.info('creating scrip file for BedMachine dataset')
            args = ['create_SCRIP_file_from_planar_rectangular_grid.py',
                    '-i', data_path + '/BedMachineGreenland-v5_edits_floodFill_extrap.nc',  # noqa
                    '-s', data_path + '/BedMachineGreenland-v5.scrip.nc',
                    '-p', 'gis-gimp', '-r', '2']
            check_call(args, logger=logger)
        if exists(data_path + '/greenland_vel_mosaic500.scrip.nc'):
            logger.info('Measures script file exists; skipping file creation')
        else:
            logger.info('creating scrip file for 2006-2010 velocity dataset')
            args = ['create_SCRIP_file_from_planar_rectangular_grid.py',
                    '-i', data_path + '/greenland_vel_mosaic500_extrap.nc',
                    '-s', data_path + '/greenland_vel_mosaic500.scrip.nc',
                    '-p', 'gis-gimp', '-r', '2']
            check_call(args, logger=logger)

        logger.info('calling set_lat_lon_fields_in_planar_grid.py')
        args = ['set_lat_lon_fields_in_planar_grid.py', '-f',
                'GIS.nc', '-p', 'gis-gimp']
        check_call(args, logger=logger)

        logger.info('creating scrip file for destination mesh')
        scrip_from_mpas('GIS.nc', 'GIS.scrip.nc')
        args = ['create_SCRIP_file_from_MPAS_mesh.py',
                '-m', 'GIS.nc',
                '-s', 'GIS.scrip.nc']
        check_call(args, logger=logger)

        # Create weight files from datasets to mesh
        if exists('BedMachine_to_MPAS_weights.nc'):
            logger.info('BedMachine_to_MPAS_weights.nc exists; skipping')
        else:
            logger.info('generating gridded dataset -> MPAS weights')
            args = ['srun', '-n', nProcs, 'ESMF_RegridWeightGen', '--source',
                    data_path + 'BedMachineGreenland-v5.scrip.nc',
                    '--destination',
                    'GIS.scrip.nc',
                    '--weight', 'BedMachine_to_MPAS_weights.nc',
                    '--method', 'conserve',
                    "-i", "-64bit_offset",
                    "--dst_regional", "--src_regional", '--netcdf4']
            check_call(args, logger=logger)

        if exists('measures_to_MPAS_weights.nc'):
            logger.info('measures_to_MPAS_weights.nc exists; skipping')
        else:
            logger.info('generating gridded dataset -> MPAS weights')
            args = ['srun', '-n', nProcs, 'ESMF_RegridWeightGen', '--source',
                    data_path + 'greenland_vel_mosaic500.scrip.nc',
                    '--destination',
                    'GIS.scrip.nc',
                    '--weight', 'measures_to_MPAS_weights.nc',
                    '--method', 'conserve',
                    "-i", "-64bit_offset", '--netcdf4',
                    "--dst_regional", "--src_regional", '--ignore_unmapped']
            check_call(args, logger=logger)

        # interpolate fields from BedMachine and Measures
        # Using conservative remapping
        logger.info('calling interpolate_to_mpasli_grid.py')
        args = ['interpolate_to_mpasli_grid.py', '-s',
                data_path + '/BedMachineGreenland-v5_edits_floodFill_extrap.nc',  # noqa
                '-d', 'GIS.nc', '-m', 'e',
                '-w', 'BedMachine_to_MPAS_weights.nc']
        check_call(args, logger=logger)

        logger.info('calling interpolate_to_mpasli_grid.py')
        args = ['interpolate_to_mpasli_grid.py', '-s',
                data_path + '/greenland_vel_mosaic500_extrap.nc',
                '-d', 'GIS.nc', '-m', 'e',
                '-w', 'measures_to_MPAS_weights.nc',
                '-v', 'observedSurfaceVelocityX',
                'observedSurfaceVelocityY',
                'observedSurfaceVelocityUncertainty']
        check_call(args, logger=logger)

        logger.info('Marking domain boundaries dirichlet')
        args = ['mark_domain_boundaries_dirichlet.py',
                '-f', 'GIS.nc']
        check_call(args, logger=logger)

        logger.info('creating graph.info')
        make_graph_file(mesh_filename=mesh_name,
                        graph_filename='graph.info')
        # Create a backup in case clean-up goes awry
        copyfile('GIS.nc', 'GIS_backup.nc')

        # Clean up: trim to iceMask and set large velocity
        # uncertainties where appropriate.
        data = netCDF4.Dataset('GIS.nc', 'r+')
        data.set_auto_mask(False)
        data.variables['thickness'][:] *= (data.variables['iceMask'][:] > 1.5)

        mask = np.logical_or(
            np.isnan(
                data.variables['observedSurfaceVelocityUncertainty'][:]),
            data.variables['thickness'][:] < 1.0)
        mask = np.logical_or(
            mask,
            data.variables['observedSurfaceVelocityUncertainty'][:] == 0.0)
        data.variables['observedSurfaceVelocityUncertainty'][0, mask[0, :]] = 1.0  # noqa

        # create region masks
        mask_filename = f'{mesh_name[:-3]}_ismip6_regionMasks.nc'
        make_region_masks(self, mesh_name, mask_filename,
                          self.cpus_per_task,
                          tags=["Greenland", "ISMIP6", "Shelf"],
                          component='ocean')

        mask_filename = f'{mesh_name[:-3]}_zwally_regionMasks.nc'
        make_region_masks(self, mesh_name, mask_filename,
                          self.cpus_per_task,
                          tags=['eastCentralGreenland',
                                'northEastGreenland',
                                'northGreenland',
                                'northWestGreenland',
                                'southEastGreenland',
                                'southGreenland',
                                'southWestGreenland',
                                'westCentralGreenland'],
                          all_tags=False)

        # Ensure basalHeatFlux is positive
        data.variables['basalHeatFlux'][:] = np.abs(
            data.variables['basalHeatFlux'][:])
        # Ensure reasonable dHdt values
        dHdt = data.variables["observedThicknessTendency"][:]
        dHdtErr = data.variables["observedThicknessTendencyUncertainty"][:]
        # Arbitrary 5% uncertainty; improve this later
        dHdtErr = np.abs(dHdt) * 0.05
        # large uncertainty where data is missing
        dHdtErr[np.abs(dHdt) > 1.0] = 1.0
        dHdt[np.abs(dHdt) > 1.0] = 0.0  # Remove ridiculous values
        data.variables["observedThicknessTendency"][:] = dHdt
        data.variables["observedThicknessTendencyUncertainty"][:] = dHdtErr

        data.close()
