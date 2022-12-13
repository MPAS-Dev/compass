from compass.step import Step

import os

import compass.ocean.tests.tides.init.dem_pixel as dem_pixel
import compass.ocean.tests.tides.init.dem_remap as dem_remap
import compass.ocean.tests.tides.init.dem_trnsf as dem_trnsf


class RemapBathymetry(Step):
    """
    A step for remapping bathymetric data onto the MPAS-Ocean mesh

    Attributes
    ----------
    plot : bool
        Whether to produce plots of remapped data
    """
    def __init__(self, test_case, mesh):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.tides.init.Init
            The test case this step belongs to

        mesh : compass.ocean.tests.tides.mesh.Mesh
            The test case that creates the mesh used by this test case
        """
        super().__init__(test_case=test_case, name='remap',
                         ntasks=1, min_tasks=1, openmp_threads=1)

        self.plot = True

        self.add_input_file(
            filename='GEBCO_2020.nc',
            target='GEBCO_2020.nc',
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

        cull_mesh_path = mesh.steps['cull_mesh'].path
        base_mesh_path = mesh.steps['base_mesh'].path

        self.add_input_file(
            filename='mesh.nc',
            work_dir_target=f'{cull_mesh_path}/culled_mesh.nc')
        self.add_input_file(
            filename='base_mesh.nc',
            work_dir_target=f'{base_mesh_path}/base_mesh.nc')

    def run(self):
        """
        Run this step of the test case
        """

        self.init_path = './'

        if not os.path.exists('RTopo_2_0_4_30sec_pixel.nc'):
            dem_pixel.rtopo_30sec(self.init_path, self.init_path)
        if not os.path.exists('GEBCO_v2020_30sec_pixel.nc'):
            dem_pixel.gebco_30sec(self.init_path, self.init_path)
        if not os.path.exists('RTopo_2_0_4_GEBCO_v2020_30sec_pixel.nc'):
            dem_pixel.rtopo_gebco_30sec(self.init_path, self.init_path)

        dem_remap.dem_remap('RTopo_2_0_4_GEBCO_v2020_30sec_pixel.nc',
                            'base_mesh.nc')
        dem_trnsf.dem_trnsf('base_mesh.nc', 'mesh.nc')
