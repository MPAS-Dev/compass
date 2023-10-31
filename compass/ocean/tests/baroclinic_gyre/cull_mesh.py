import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull

from compass.step import Step


class CullMesh(Step):
    """
    Cull a global mesh to only a signle basin
    Attributes
    ----------
    """
    def __init__(self, test_case):
        """
        Create the step
        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='cull_mesh')

        self.add_input_file(
            filename='base_mesh.nc',
            target='../base_mesh/base_mesh.nc')

        for file in ['culled_mesh.nc', 'culled_graph.info']:
            self.add_output_file(file)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        section = config['baroclinic_gyre']
        logger = self.logger
        ds_mesh = xr.open_dataset('base_mesh.nc')
        ds_mask = xr.Dataset()

        lon = np.rad2deg(ds_mesh.lonCell.values)
        lat = np.rad2deg(ds_mesh.latCell.values)
        lon_min = section.getfloat('lon_min')
        lon_max = section.getfloat('lon_max')
        lat_min = section.getfloat('lat_min')
        lat_max = section.getfloat('lat_max')

        mask = np.logical_and(
            np.logical_and(lon >= lon_min, lon <= lon_max),
            np.logical_and(lat >= lat_min, lat <= lat_max))

        n_cells = ds_mesh.sizes['nCells']
        ds_mask['regionCellMasks'] = (('nCells', 'nRegions'),
                                      mask.astype(int).reshape(n_cells, 1))

        ds_mesh = cull(ds_mesh, dsInverse=ds_mask, logger=logger)
        ds_mesh = convert(ds_mesh, graphInfoFileName='culled_graph.info',
                          logger=logger)

        write_netcdf(ds_mesh, 'culled_mesh.nc')
