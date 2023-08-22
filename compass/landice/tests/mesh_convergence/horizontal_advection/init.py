import numpy
import xarray
from mpas_tools.io import write_netcdf

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

        section = config['horizontal_advection']
        ice_thickness = section.getfloat('ice_thickness')
        bed_elevation = section.getfloat('bed_elevation')
        x_center = 1e3 * section.getfloat('x_center')
        y_center = 1e3 * section.getfloat('y_center')
        advect_x = section.getboolean('advect_x')
        advect_y = section.getboolean('advect_y')
        gaussian_width = 1e3 * section.getfloat('gaussian_width')
        nVertLevels = section.getint('vert_levels')

        section = config['mesh_convergence']
        duration = section.getfloat('duration') * 3600.0 * 24.0 * 365.0

        ds = xarray.open_dataset('mesh.nc')
        xCell = ds.xCell
        yCell = ds.yCell

        if advect_x:
            x_vel = ds.attrs['x_period'] / duration
        else:
            x_vel = 0.

        if advect_y:
            y_vel = ds.attrs['y_period'] / duration
        else:
            y_vel = 0.

        layerThicknessFractions = xarray.DataArray(
            data=1.0 / nVertLevels * numpy.ones((nVertLevels,)),
            dims=['nVertLevels'])

        thickness = ice_thickness * xarray.ones_like(xCell)
        thickness = thickness.expand_dims(dim='Time', axis=0)

        bedTopography = bed_elevation * xarray.ones_like(thickness)

        uReconstructX = x_vel * xarray.ones_like(xCell)
        uReconstructX = uReconstructX.expand_dims(dim={"nVertInterfaces":
                                                       nVertLevels + 1},
                                                  axis=1)
        uReconstructX = uReconstructX.expand_dims(dim='Time', axis=0)

        uReconstructY = y_vel * xarray.ones_like(xCell)
        uReconstructY = uReconstructY.expand_dims(dim={"nVertInterfaces":
                                                       nVertLevels + 1},
                                                  axis=1)
        uReconstructY = uReconstructY.expand_dims(dim='Time', axis=0)

        dist_sq = (xCell - x_center)**2 + (yCell - y_center)**2

        passiveTracer = numpy.exp(-0.5 * dist_sq / gaussian_width**2)
        passiveTracer = passiveTracer.expand_dims(dim='Time', axis=0)

        ds['layerThicknessFractions'] = layerThicknessFractions
        ds['thickness'] = thickness
        ds['bedTopography'] = bedTopography
        ds['uReconstructX'] = uReconstructX
        ds['uReconstructY'] = uReconstructY
        ds['passiveTracer'] = passiveTracer

        write_netcdf(ds, 'initial_state.nc')
