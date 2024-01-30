from shutil import copyfile

import netCDF4

from compass.landice.mesh import (
    build_cell_width,
    build_mali_mesh,
    make_region_masks,
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
        section_name = 'mesh'

        logger.info('calling build_cell_width')
        cell_width, x1, y1, geom_points, geom_edges, floodFillMask = \
            build_cell_width(
                self, section_name=section_name,
                gridded_dataset='antarctica_8km_2024_01_29.nc')

        # Apply floodFillMask to thickness field to help with culling
        copyfile('antarctica_8km_2024_01_29.nc',
                 'antarctica_8km_2024_01_29_floodFillMask.nc')
        gg = netCDF4.Dataset('antarctica_8km_2024_01_29_floodFillMask.nc',
                             'r+')
        gg.variables['thk'][0, :, :] *= floodFillMask
        gg.variables['vx'][0, :, :] *= floodFillMask
        gg.variables['vy'][0, :, :] *= floodFillMask
        gg.close()

        build_mali_mesh(
            self, cell_width, x1, y1, geom_points, geom_edges,
            mesh_name=self.mesh_filename, section_name=section_name,
            gridded_dataset='antarctica_8km_2024_01_29_floodFillMask.nc',
            projection='ais-bedmap2', geojson_file=None)

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
