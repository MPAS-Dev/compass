import xarray

from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import cull

from compass.step import Step


class CulledMesh(Step):
    """
    A step for culling the base mesh for SOMA test cases
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='culled_mesh')

        self.add_input_file(filename='masked_initial_state.nc',
                            target='../init_on_base_mesh/initial_state.nc')

        for file in ['culled_mesh.nc', 'culled_graph.info']:
            self.add_output_file(filename=file)

    def run(self):
        """
        Run this step of the test case
        """
        ds_mesh = cull(xarray.open_dataset('masked_initial_state.nc'),
                       graphInfoFileName='culled_graph.info',
                       logger=self.logger)
        write_netcdf(ds_mesh, 'culled_mesh.nc')
