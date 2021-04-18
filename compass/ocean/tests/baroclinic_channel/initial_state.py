import xarray
import numpy

from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull

from compass.ocean.vertical import generate_grid
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for baroclinic channel
    test cases

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
        super().__init__(test_case=test_case, name='initial_state')
        self.resolution = resolution

        for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info',
                     'ocean.nc']:
            self.add_output_file(file)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger

        section = config['baroclinic_channel']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=False,
                                      nonperiodic_y=True)
        write_netcdf(dsMesh, 'base_mesh.nc')

        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, graphInfoFileName='culled_graph.info',
                         logger=logger)
        write_netcdf(dsMesh, 'culled_mesh.nc')

        section = config['baroclinic_channel']
        use_distances = section.getboolean('use_distances')
        gradient_width_dist = section.getfloat('gradient_width_dist')
        gradient_width_frac = section.getfloat('gradient_width_frac')
        bottom_temperature = section.getfloat('bottom_temperature')
        surface_temperature = section.getfloat('surface_temperature')
        temperature_difference = section.getfloat('temperature_difference')
        salinity = section.getfloat('salinity')
        coriolis_parameter = section.getfloat('coriolis_parameter')

        ds = dsMesh.copy()

        interfaces = generate_grid(config=config)

        bottom_depth = interfaces[-1]
        vert_levels = len(interfaces) - 1

        ds['refBottomDepth'] = ('nVertLevels', interfaces[1:])
        ds['refZMid'] = ('nVertLevels', -0.5 * (interfaces[1:] + interfaces[0:-1]))
        ds['vertCoordMovementWeights'] = xarray.ones_like(ds.refBottomDepth)

        xCell = ds.xCell
        yCell = ds.yCell

        xMin = xCell.min().values
        xMax = xCell.max().values
        yMin = yCell.min().values
        yMax = yCell.max().values

        yMid = 0.5*(yMin + yMax)
        xPerturbMin = xMin + 4.0 * (xMax - xMin) / 6.0
        xPerturbMax = xMin + 5.0 * (xMax - xMin) / 6.0

        if use_distances:
            perturbationWidth = gradient_width_dist
        else:
            perturbationWidth = (yMax - yMin) * gradient_width_frac

        yOffset = perturbationWidth * numpy.sin(
            6.0 * numpy.pi * (xCell - xMin) / (xMax - xMin))

        temp_vert = (bottom_temperature +
                     (surface_temperature - bottom_temperature) *
                     ((ds.refZMid + bottom_depth) / bottom_depth))

        frac = xarray.where(yCell < yMid - yOffset, 1., 0.)

        mask = numpy.logical_and(yCell >= yMid - yOffset,
                                 yCell < yMid - yOffset + perturbationWidth)
        frac = xarray.where(mask,
                            1. - (yCell - (yMid - yOffset)) / perturbationWidth,
                            frac)

        temperature = temp_vert - temperature_difference * frac
        temperature = temperature.transpose('nCells', 'nVertLevels')

        # Determine yOffset for 3rd crest in sin wave
        yOffset = 0.5 * perturbationWidth * numpy.sin(
            numpy.pi * (xCell - xPerturbMin) / (xPerturbMax - xPerturbMin))

        mask = numpy.logical_and(
            numpy.logical_and(yCell >= yMid - yOffset - 0.5 * perturbationWidth,
                              yCell <= yMid - yOffset + 0.5 * perturbationWidth),
            numpy.logical_and(xCell >= xPerturbMin,
                              xCell <= xPerturbMax))

        temperature = (temperature +
                       mask * 0.3 * (1. - ((yCell - (yMid - yOffset)) /
                                           (0.5 * perturbationWidth))))

        temperature = temperature.expand_dims(dim='Time', axis=0)

        layerThickness = xarray.DataArray(data=interfaces[1:] - interfaces[0:-1],
                                          dims='nVertLevels')
        _, layerThickness = xarray.broadcast(xCell, layerThickness)
        layerThickness = layerThickness.transpose('nCells', 'nVertLevels')
        layerThickness = layerThickness.expand_dims(dim='Time', axis=0)

        normalVelocity = xarray.zeros_like(ds.xEdge)
        normalVelocity, _ = xarray.broadcast(normalVelocity, ds.refBottomDepth)
        normalVelocity = normalVelocity.transpose('nEdges', 'nVertLevels')
        normalVelocity = normalVelocity.expand_dims(dim='Time', axis=0)

        ds['temperature'] = temperature
        ds['salinity'] = salinity * xarray.ones_like(temperature)
        ds['normalVelocity'] = normalVelocity
        ds['layerThickness'] = layerThickness
        ds['restingThickness'] = layerThickness
        ds['bottomDepth'] = bottom_depth * xarray.ones_like(xCell)
        ds['maxLevelCell'] = vert_levels * xarray.ones_like(xCell, dtype=int)
        ds['fCell'] = coriolis_parameter * xarray.ones_like(xCell)
        ds['fEdge'] = coriolis_parameter * xarray.ones_like(ds.xEdge)
        ds['fVertex'] = coriolis_parameter * xarray.ones_like(ds.xVertex)

        write_netcdf(ds, 'ocean.nc')
