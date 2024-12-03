import netCDF4 as nc4

import compass.ocean.tests.tides.dem.dem_remap as dem_remap
import compass.ocean.tests.tides.dem.dem_trnsf as dem_trnsf
from compass.step import Step


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

        cull_mesh_path = mesh.steps['cull_mesh'].path
        base_mesh_path = mesh.steps['base_mesh'].path
        pixel_path = mesh.steps['pixel'].path

        self.add_input_file(
            filename='mesh.nc',
            work_dir_target=f'{cull_mesh_path}/culled_mesh.nc')
        self.add_input_file(
            filename='base_mesh.nc',
            work_dir_target=f'{base_mesh_path}/base_mesh.nc')
        pixel_file = f'{pixel_path}/RTopo_2_0_4_GEBCO_v2023_30sec_pixel.nc'
        self.add_input_file(
            filename='bathy.nc',
            work_dir_target=pixel_file)

    def run(self):
        """
        Run this step of the test case
        """
        dem_remap.dem_remap('bathy.nc',
                            'base_mesh.nc')
        dem_trnsf.dem_trnsf('base_mesh.nc', 'mesh.nc')

        # Create new NetCDF variables in mesh file, if necessary
        nc_mesh = nc4.Dataset('mesh.nc', 'r+')
        nc_vars = nc_mesh.variables.keys()
        if 'bottomDepthObserved' not in nc_vars:
            nc_mesh.createVariable('bottomDepthObserved', 'f8', ('nCells'))

        # Write to mesh file
        nc_mesh.variables['bottomDepthObserved'][:] = \
            nc_mesh.variables['bed_elevation'][:]
        nc_mesh.close()
