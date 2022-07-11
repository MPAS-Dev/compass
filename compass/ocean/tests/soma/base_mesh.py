import xarray

from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert

from compass.step import Step


class BaseMesh(Step):
    """
    A step for converting a base mesh to MPAS format for SOMA test cases

    Attributes
    ----------
    resolution : str
        The resolution of the test case
    """

    def __init__(self, test_case, resolution):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : str
            The resolution of the test case
        """
        self.resolution = resolution

        super().__init__(test_case=test_case, name='base_mesh')

        mesh_filenames = {'32km': 'SOMA_32km_grid.161202.nc',
                          '16km': 'SOMA_16km_grid.161202.nc',
                          '8km': 'SOMA_8km_grid.161202.nc',
                          '4km': 'SOMA_4km_grid.161202.nc'}
        if resolution not in mesh_filenames:
            raise ValueError(f'Unexpected SOMA resolution: {resolution}')

        self.add_input_file(filename='base_mesh.nc',
                            target=mesh_filenames[resolution],
                            database='mesh_database')

        for file in ['mesh.nc', 'base_graph.info']:
            self.add_output_file(filename=file)

    def run(self):
        """
        Convert the base mesh to MPAS format
        """
        ds_mesh = convert(xarray.open_dataset('base_mesh.nc'),
                          graphInfoFileName='base_graph.info',
                          logger=self.logger)
        write_netcdf(ds_mesh, 'mesh.nc')
