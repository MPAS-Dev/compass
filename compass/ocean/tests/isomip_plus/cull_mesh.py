import xarray as xr

from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull

from compass.step import Step
from compass.ocean.tests.isomip_plus.geom import \
    define_thin_film_mask_step1, interpolate_ocean_mask


class CullMesh(Step):
    """
    Cull an ISOMIP+ mesh to only include floating cells (possibly including
    a thin-film region for grounding-line retreat).

    Attributes
    ----------
    thin_film_present: bool
        Whether the run includes a thin film below grounded ice

    planar : bool
        Whether the test case runs on a planar or a spherical mesh
    """
    def __init__(self, test_case, thin_film_present, planar):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        thin_film_present: bool
            Whether the run includes a thin film below grounded ice

        planar : bool
            Whether the test case runs on a planar or a spherical mesh
        """
        super().__init__(test_case=test_case, name='cull_mesh')
        self.thin_film_present = thin_film_present
        self.planar = planar

        self.add_input_file(
            filename='input_geometry_processed.nc',
            target='../process_geom/input_geometry_processed.nc')

        if planar:
            self.add_input_file(
                filename='base_mesh.nc',
                target='../planar_mesh/base_mesh.nc')
        else:
            self.add_input_file(
                filename='base_mesh.nc',
                target='../spherical_mesh/base_mesh_with_xy.nc')

        for file in ['culled_mesh.nc', 'culled_graph.info']:
            self.add_output_file(file)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger
        section = config['isomip_plus']
        min_ocean_fraction = section.getfloat('min_ocean_fraction')
        thin_film_present = self.thin_film_present
        ds_mesh = xr.open_dataset('base_mesh.nc')
        ds_geom = xr.open_dataset('input_geometry_processed.nc')

        if thin_film_present:
            ds_mask = define_thin_film_mask_step1(ds_mesh, ds_geom)
        else:
            ds_mask = interpolate_ocean_mask(ds_mesh, ds_geom, min_ocean_fraction)
        ds_mesh = cull(ds_mesh, dsInverse=ds_mask, logger=logger)
        ds_mesh.attrs['is_periodic'] = 'NO'

        ds_mesh = convert(ds_mesh, graphInfoFileName='culled_graph.info',
                          logger=logger)
        write_netcdf(ds_mesh, 'culled_mesh.nc')
