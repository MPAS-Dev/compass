import xarray
import numpy

from mpas_tools.io import write_netcdf

from compass.ocean.vertical import init_vertical_coord
from compass.ocean.tests.planar_convergence.conv_init import ConvInit


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
        temperature = section.getfloat('temperature')
        salinity = section.getfloat('salinity')
        x_center = 1e3*section.getfloat('x_center')
        y_center = 1e3*section.getfloat('y_center')
        advect_x = section.getboolean('advect_x')
        advect_y = section.getboolean('advect_y')
        gaussian_width = 1e3*section.getfloat('gaussian_width')

        section = config['planar_convergence']
        duration = 3600.*section.getfloat('duration')
        dt_1km = section.getint('dt_1km')
        resolution = float(self.resolution)
        dt = dt_1km * resolution
        dc = resolution*1e3

        ds = xarray.open_dataset('mesh.nc')
        xCell = ds.xCell
        yCell = ds.yCell

        bottom_depth = config.getfloat('vertical_grid', 'bottom_depth')

        ds['bottomDepth'] = bottom_depth * xarray.ones_like(xCell)
        ds['ssh'] = xarray.zeros_like(xCell)

        init_vertical_coord(config, ds)

        if advect_x:
            x_vel = ds.attrs['x_period']/duration
            x_cfl = x_vel*dt/dc
            print(f'x_cfl: {x_cfl}')
        else:
            x_vel = 0.

        if advect_y:
            y_vel = ds.attrs['y_period']/duration
            y_cfl = y_vel*dt/dc
            print(f'y_cfl: {y_cfl}')
        else:
            y_vel = 0.

        temperature = temperature*xarray.ones_like(xCell)
        temperature, _ = xarray.broadcast(temperature, ds.refBottomDepth)
        temperature = temperature.transpose('nCells', 'nVertLevels')
        temperature = temperature.expand_dims(dim='Time', axis=0)

        salinity = salinity*xarray.ones_like(temperature)

        angleEdge = ds.angleEdge
        normalVelocity = (numpy.cos(angleEdge) * x_vel +
                          numpy.sin(angleEdge) * y_vel)
        normalVelocity, _ = xarray.broadcast(normalVelocity, ds.refBottomDepth)
        normalVelocity = normalVelocity.transpose('nEdges', 'nVertLevels')
        normalVelocity = normalVelocity.expand_dims(dim='Time', axis=0)

        dist_sq = (xCell - x_center)**2 + (yCell - y_center)**2

        tracer1 = numpy.exp(-0.5*dist_sq/gaussian_width**2)
        tracer1, _ = xarray.broadcast(tracer1, ds.refBottomDepth)
        tracer1 = tracer1.transpose('nCells', 'nVertLevels')
        tracer1 = tracer1.expand_dims(dim='Time', axis=0)

        ds['temperature'] = temperature
        ds['salinity'] = salinity * xarray.ones_like(temperature)
        ds['normalVelocity'] = normalVelocity
        ds['fCell'] = xarray.zeros_like(xCell)
        ds['fEdge'] = xarray.zeros_like(ds.xEdge)
        ds['fVertex'] = xarray.zeros_like(ds.xVertex)
        ds['tracer1'] = tracer1

        write_netcdf(ds, 'initial_state.nc')
