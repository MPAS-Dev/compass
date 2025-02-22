import os

import numpy as np
import xarray as xr
from mpas_tools.scrip.from_mpas import scrip_from_mpas

from compass.landice.mesh import (
    build_cell_width,
    build_mali_mesh,
    clean_up_after_interp,
    interp_gridded2mali,
    make_region_masks,
)
from compass.model import make_graph_file
from compass.step import Step


class Mesh(Step):
    """
    A step for creating a mesh and initial condition for greenland test cases

    Attributes
    ----------
    mesh_filename : str
        File name of the MALI mesh
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        """
        super().__init__(test_case=test_case, name='mesh', cpus_per_task=128,
                         min_cpus_per_task=1)

        # output files
        self.mesh_filename = 'GIS.nc'
        self.add_output_file(filename='graph.info')
        self.add_output_file(filename=self.mesh_filename)
        self.add_output_file(
            filename=f'{self.mesh_filename[:-3]}_ismip6_regionMasks.nc')
        self.add_output_file(
            filename=f'{self.mesh_filename[:-3]}_zwally_regionMasks.nc')
        # input files
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
        config = self.config

        section_gis = config['greenland']

        parallel_executable = config.get('parallel', 'parallel_executable')
        nProcs = section_gis.get('nProcs')
        src_proj = section_gis.get("src_proj")
        data_path = section_gis.get('data_path')
        measures_filename = section_gis.get("measures_filename")
        bedmachine_filename = section_gis.get("bedmachine_filename")

        measures_dataset = os.path.join(data_path, measures_filename)
        bedmachine_dataset = os.path.join(data_path, bedmachine_filename)

        section_name = 'mesh'

        source_gridded_dataset_1km = 'greenland_1km_2024_01_29.epsg3413.icesheetonly.nc'  # noqa: E501
        source_gridded_dataset_2km = 'greenland_2km_2024_01_29.epsg3413.nc'

        logger.info('calling build_cell_width')
        cell_width, x1, y1, geom_points, geom_edges, floodMask = \
            build_cell_width(
                self, section_name=section_name,
                gridded_dataset=source_gridded_dataset_2km,
                flood_fill_start=[100, 700])

        # Now build the base mesh and perform the standard interpolation
        build_mali_mesh(
            self, cell_width, x1, y1, geom_points, geom_edges,
            mesh_name=self.mesh_filename, section_name=section_name,
            gridded_dataset=source_gridded_dataset_1km,
            projection=src_proj, geojson_file=None)

        # Create scrip file for the newly generated mesh
        logger.info('creating scrip file for destination mesh')
        dst_scrip_file = f"{self.mesh_filename.split('.')[:-1][0]}_scrip.nc"
        scrip_from_mpas(self.mesh_filename, dst_scrip_file)

        # Now perform bespoke interpolation of geometry and velocity data
        # from their respective sources
        interp_gridded2mali(self, bedmachine_dataset, dst_scrip_file,
                            parallel_executable, nProcs,
                            self.mesh_filename, src_proj, variables="all")

        # only interpolate a subset of MEaSUREs variables onto the MALI mesh
        measures_vars = ['observedSurfaceVelocityX',
                         'observedSurfaceVelocityY',
                         'observedSurfaceVelocityUncertainty']
        interp_gridded2mali(self, measures_dataset, dst_scrip_file,
                            parallel_executable, nProcs,
                            self.mesh_filename, src_proj,
                            variables=measures_vars)

        # perform some final cleanup details
        clean_up_after_interp(self.mesh_filename)

        # create graph file
        logger.info('creating graph.info')
        make_graph_file(mesh_filename=self.mesh_filename,
                        graph_filename='graph.info')

        # create region masks
        mask_filename = f'{self.mesh_filename[:-3]}_ismip6_regionMasks.nc'
        make_region_masks(self, self.mesh_filename, mask_filename,
                          self.cpus_per_task,
                          tags=["Greenland", "ISMIP6", "Shelf"],
                          component='ocean')

        mask_filename = f'{self.mesh_filename[:-3]}_zwally_regionMasks.nc'
        make_region_masks(self, self.mesh_filename, mask_filename,
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

        # Do some final validation of the mesh
        ds = xr.open_dataset(self.mesh_filename)
        # Ensure basalHeatFlux is positive
        ds["basalHeatFlux"] = np.abs(ds.basalHeatFlux)
        # Ensure reasonable dHdt values
        dHdt = ds["observedThicknessTendency"]
        # Arbitrary 5% uncertainty; improve this later
        dHdtErr = np.abs(dHdt) * 0.05
        # Use threshold of |dHdt| > 1.0 to determine invalid data
        mask = np.abs(dHdt) > 1.0
        # Assign very large uncertainty where data is missing
        dHdtErr = dHdtErr.where(~mask, 1.0)
        # Remove ridiculous values
        dHdt = dHdt.where(~mask, 0.0)
        # Put the updated fields back in the dataset
        ds["observedThicknessTendency"] = dHdt
        ds["observedThicknessTendencyUncertainty"] = dHdtErr
        # Write the data to disk
        ds.to_netcdf(self.mesh_filename, 'a')
