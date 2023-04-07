from shutil import copyfile

import matplotlib.pyplot as plt
import mpas_tools
import netCDF4
from geometric_features import FeatureCollection, GeometricFeatures
from mpas_tools.logging import check_call

from compass.landice.mesh import build_cell_width, build_MALI_mesh
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
        super().__init__(test_case=test_case, name='mesh', cpus_per_task=64,
                         min_cpus_per_task=1)

        self.mesh_filename = 'Antarctica.nc'
        self.add_output_file(filename='graph.info')
        self.add_output_file(filename=self.mesh_filename)
        self.add_output_file(filename=f'{self.mesh_filename[:-3]}_'
                                      f'imbie_regionMasks.nc')
        self.add_output_file(filename=f'{self.mesh_filename[:-3]}_'
                                      f'ismip6_regionMasks.nc')
        self.add_input_file(
            filename='antarctica_8km_2020_10_20.nc',
            target='antarctica_8km_2020_10_20.nc',
            database='')

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        section_name = 'antarctica'

        logger.info('calling build_cell_width')
        cell_width, x1, y1, geom_points, geom_edges, floodFillMask = \
            build_cell_width(
                self, section_name='antarctica',
                gridded_dataset='antarctica_8km_2020_10_20.nc')

        # Apply floodFillMask to thickness field to help with culling
        copyfile('antarctica_8km_2020_10_20.nc',
                 'antarctica_8km_2020_10_20_floodFillMask.nc')
        gg = netCDF4.Dataset('antarctica_8km_2020_10_20_floodFillMask.nc',
                             'r+')
        gg.variables['thk'][0, :, :] *= floodFillMask
        gg.variables['vx'][0, :, :] *= floodFillMask
        gg.variables['vy'][0, :, :] *= floodFillMask
        gg.close()

        build_MALI_mesh(
            self, cell_width, x1, y1, geom_points, geom_edges,
            mesh_name=self.mesh_filename, section_name=section_name,
            gridded_dataset='antarctica_8km_2020_10_20.nc',
            projection='ais-bedmap2', geojson_file=None)

        logger.info('creating graph.info')
        make_graph_file(mesh_filename=self.mesh_filename,
                        graph_filename='graph.info')

        # create a region mask
        mask_filename = f'{self.mesh_filename[:-3]}_imbie_regionMasks.nc'
        self._make_region_masks(self.mesh_filename, mask_filename,
                                self.cpus_per_task,
                                tags=['EastAntarcticaIMBIE',
                                      'WestAntarcticaIMBIE',
                                      'AntarcticPeninsulaIMBIE'])

        mask_filename = f'{self.mesh_filename[:-3]}_ismip6_regionMasks.nc'
        self._make_region_masks(self.mesh_filename, mask_filename,
                                self.cpus_per_task,
                                tags=['ISMIP6_Basin'])

    def _make_region_masks(self, mesh_filename, mask_filename, cores, tags):
        logger = self.logger
        gf = GeometricFeatures()
        fcMask = FeatureCollection()

        for tag in tags:
            fc = gf.read(componentName='landice', objectType='region',
                         tags=[tag])
            fc.plot('southpole')
            plt.savefig(f'plot_basins_{tag}.png')
            fcMask.merge(fc)

        geojson_filename = 'regionMask.geojson'
        fcMask.to_geojson(geojson_filename)

        # these defaults may have been updated from config options -- pass them
        # along to the subprocess
        netcdf_format = mpas_tools.io.default_format
        netcdf_engine = mpas_tools.io.default_engine

        args = ['compute_mpas_region_masks',
                '-m', mesh_filename,
                '-g', geojson_filename,
                '-o', mask_filename,
                '-t', 'cell',
                '--process_count', f'{cores}',
                '--format', netcdf_format,
                '--engine', netcdf_engine]
        check_call(args, logger=logger)
