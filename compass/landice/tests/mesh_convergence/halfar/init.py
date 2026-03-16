import numpy
import xarray
from mpas_tools.io import write_netcdf

from compass.landice.tests.dome.setup_mesh import setup_dome_initial_conditions
from compass.landice.tests.mesh_convergence.conv_init import ConvInit


class Init(ConvInit):
    """
    A step for creating an initial_condition for advection convergence test
    case
    """
    def __init__(self, test_case, resolution):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : int
            The resolution of the test case
        """
        super().__init__(test_case=test_case, resolution=resolution)
        self.add_output_file('initial_state.nc')

    def run(self):
        """
        Run this step of the test case
        """
        # create the mesh and graph.info
        super().run()

        config = self.config
        logger = self.logger
        filename = 'initial_state.nc'

        section = config['halfar']
        nVertLevels = section.getint('vert_levels')

        ds = xarray.open_dataset('mesh.nc')
        xCell = ds.xCell

        layerThicknessFractions = xarray.DataArray(
            data=1.0 / nVertLevels * numpy.ones((nVertLevels,)),
            dims=['nVertLevels'])
        ds['layerThicknessFractions'] = layerThicknessFractions
        thickness = xarray.zeros_like(xCell)
        thickness = thickness.expand_dims(dim='Time', axis=0)
        ds['thickness'] = thickness
        ds['bedTopography'] = xarray.zeros_like(thickness)
        ds['sfcMassBal'] = xarray.zeros_like(thickness)
        write_netcdf(ds, filename)

        setup_dome_initial_conditions(config, logger, filename)
