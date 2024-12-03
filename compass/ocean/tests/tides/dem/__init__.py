import os

import compass.ocean.tests.tides.dem.dem_pixel as dem_pixel
from compass.step import Step


class CreatePixelFile(Step):
    """
    A step for creating a pixel file for creating MPAS meshes

    Attributes
    ----------
    """

    def __init__(self, test_case, resolution='30sec'):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.tides.init.Init
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='pixel',
                         ntasks=1, min_tasks=1, openmp_threads=1)

        self.add_input_file(
            filename='GEBCO_2023_sub_ice_topo.nc',
            target='GEBCO_2023_sub_ice_topo.nc',
            database='bathymetry_database')

        self.add_input_file(
            filename='RTopo-2.0.4_30sec_bedrock_topography.nc',
            target='RTopo-2.0.4_30sec_bedrock_topography.nc',
            database='bathymetry_database')

        self.add_input_file(
            filename='RTopo-2.0.4_30sec_surface_elevation.nc',
            target='RTopo-2.0.4_30sec_surface_elevation.nc',
            database='bathymetry_database')

        self.add_input_file(
            filename='RTopo-2.0.4_30sec_ice_base_topography.nc',
            target='RTopo-2.0.4_30sec_ice_base_topography.nc',
            database='bathymetry_database')

        self.resolution = resolution

    def run(self):
        """
        Run this step of the test case
        """

        self.init_path = './'

        if self.resolution == '30sec':
            if not os.path.exists('RTopo_2_0_4_30sec_pixel.nc'):
                dem_pixel.rtopo_30sec(self.init_path, self.init_path)
            if not os.path.exists('GEBCO_v2023_30sec_pixel.nc'):
                dem_pixel.gebco_30sec(self.init_path, self.init_path)
            if not os.path.exists('RTopo_2_0_4_GEBCO_v2023_30sec_pixel.nc'):
                dem_pixel.rtopo_gebco_30sec(self.init_path, self.init_path)
        elif self.resolution == '15sec':
            if not os.path.exists('RTopo_2_0_4_15sec_pixel.nc'):
                dem_pixel.rtopo_15sec(self.init_path, self.init_path)
            if not os.path.exists('GEBCO_v2023_15sec_pixel.nc'):
                dem_pixel.gebco_15sec(self.init_path, self.init_path)
            if not os.path.exists('RTopo_2_0_4_GEBCO_v2023_15sec_pixel.nc'):
                dem_pixel.rtopo_gebco_15sec(self.init_path, self.init_path)
