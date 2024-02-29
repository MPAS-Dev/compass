import os

import netCDF4
from mpas_tools.logging import check_call
from mpas_tools.scrip.from_mpas import scrip_from_mpas

from compass.landice.mesh import (
    add_bedmachine_thk_to_ais_gridded_data,
    build_cell_width,
    build_mali_mesh,
    clean_up_after_interp,
    interp_ais_bedmachine,
    interp_ais_measures,
    make_region_masks,
    preprocess_ais_data,
)
from compass.model import make_graph_file
from compass.step import Step


class Mesh(Step):
    """
    A step for creating a mesh and initial condition for Antarctica test cases

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

        self.mesh_filename = 'Antarctica.nc'
        self.add_output_file(filename='graph.info')
        self.add_output_file(filename=self.mesh_filename)
        self.add_output_file(filename=f'{self.mesh_filename[:-3]}_'
                                      f'imbie_regionMasks.nc')
        self.add_output_file(filename=f'{self.mesh_filename[:-3]}_'
                                      f'ismip6_regionMasks.nc')
        self.add_input_file(
            filename='antarctica_8km_2024_01_29.nc',
            target='antarctica_8km_2024_01_29.nc',
            database='')

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        section_ais = config['antarctica']
        data_path = section_ais.get('data_path')
        nProcs = section_ais.get('nProcs')

        section_name = 'mesh'

        source_gridded_dataset = 'antarctica_8km_2024_01_29.nc'
        bedmachine_path = os.path.join(
            data_path,
            'BedMachineAntarctica_2020-07-15_v02_edits_floodFill_extrap_fillVostok.nc')  # noqa

        bm_updated_gridded_dataset = add_bedmachine_thk_to_ais_gridded_data(
            self, source_gridded_dataset, bedmachine_path)
        logger.info('calling build_cell_width')
        cell_width, x1, y1, geom_points, geom_edges, floodFillMask = \
            build_cell_width(
                self, section_name=section_name,
                gridded_dataset=bm_updated_gridded_dataset)

        # Preprocess the gridded AIS source datasets to work
        # with the rest of the workflow
        logger.info('calling preprocess_ais_data')
        preprocessed_gridded_dataset = preprocess_ais_data(
            self, bm_updated_gridded_dataset, floodFillMask)

        # Now build the base mesh and perform the standard interpolation
        build_mali_mesh(
            self, cell_width, x1, y1, geom_points, geom_edges,
            mesh_name=self.mesh_filename, section_name=section_name,
            gridded_dataset=bm_updated_gridded_dataset,
            projection='ais-bedmap2', geojson_file=None)

        # Now that we have base mesh with standard interpolation
        # perform advanced interpolation for specific fields
        # that require more careful treatment

        # Add iceMask for later trimming if not already in file.
        # It should be automatically added as of MPAS-Tools commit
        # df90de2c434ed24bbbaf9ca353c2a91de1140654
        # Aug 8, 2022, but safest to double check here.
        data = netCDF4.Dataset(self.mesh_filename, 'r+')
        if 'iceMask' not in data.variables:
            data.createVariable('iceMask', 'f', ('Time', 'nCells'))
            data.variables['iceMask'][:] = 0.
        data.close()

        # interpolate fields from composite dataset
        # Note: this was already done in build_mali_mesh() using
        # bilinear interpolation.  Redoing it here again is likely
        # not needed.  Also, it should be assessed if bilinear or
        # barycentric used here is preferred for this application.
        # Current thinking is they are both equally appropriate.
        logger.info('calling interpolate_to_mpasli_grid.py')
        args = ['interpolate_to_mpasli_grid.py', '-s',
                preprocessed_gridded_dataset,
                '-d',
                self.mesh_filename,
                '-m', 'd', '-v',
                'floatingBasalMassBal', 'basalHeatFlux', 'sfcMassBal',
                'surfaceAirTemperature', 'observedThicknessTendency',
                'observedThicknessTendencyUncertainty', 'thickness']
        check_call(args, logger=logger)

        # Create scrip file for the newly generated mesh
        logger.info('creating scrip file for destination mesh')
        dst_scrip_file = f"{self.mesh_filename.split('.')[:-1][0]}_scrip.nc"
        scrip_from_mpas(self.mesh_filename, dst_scrip_file)

        # Now perform bespoke interpolation of geometry and velocity data
        # from their respective sources
        interp_ais_bedmachine(self, data_path, dst_scrip_file, nProcs,
                              self.mesh_filename)
        interp_ais_measures(self, data_path, dst_scrip_file, nProcs,
                            self.mesh_filename)

        # perform some final cleanup details
        clean_up_after_interp(self.mesh_filename)

        # create graph file
        logger.info('creating graph.info')
        make_graph_file(mesh_filename=self.mesh_filename,
                        graph_filename='graph.info')

        # create a region mask
        mask_filename = f'{self.mesh_filename[:-3]}_imbie_regionMasks.nc'
        make_region_masks(self, self.mesh_filename, mask_filename,
                          self.cpus_per_task,
                          tags=['EastAntarcticaIMBIE',
                                'WestAntarcticaIMBIE',
                                'AntarcticPeninsulaIMBIE'])

        mask_filename = f'{self.mesh_filename[:-3]}_ismip6_regionMasks.nc'
        make_region_masks(self, self.mesh_filename, mask_filename,
                          self.cpus_per_task,
                          tags=['ISMIP6_Basin'])
