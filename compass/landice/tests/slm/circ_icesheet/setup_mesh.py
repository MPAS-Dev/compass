import numpy
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh
from netCDF4 import Dataset as NetCDFFile

from compass.model import make_graph_file
from compass.step import Step


class SetupMesh(Step):
    """
    A step for creating a mesh with initial conditions plus forcing file for
    circular icesheet test

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
        super().__init__(test_case=test_case, name='setup_mesh')

        self.add_output_file(filename='graph.info')
        self.add_output_file(filename='landice_grid.nc')

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        section = config['dome']

        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc,
                                      nonperiodic_x=True,
                                      nonperiodic_y=True)
        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, logger=logger)
        write_netcdf(dsMesh, 'mpas_grid.nc')

        levels = 3
        args = ['create_landice_grid_from_generic_MPAS_grid.py',
                '-i', 'mpas_grid.nc',
                '-o', 'landice_grid.nc',
                '-l', levels]

        check_call(args, logger)

        make_graph_file(mesh_filename='landice_grid.nc',
                        graph_filename='graph.info')

        _setup_circsheet_initial_conditions(config, logger,
                                            filename='landice_grid.nc')


def _setup_circsheet_initial_conditions(config, logger, filename):
    """
    Create initial condition for circular ice sheet

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step

    filename : str
        file to setup dome
    """
    section = config['dome']
    dome_type = section.get('dome_type')
    put_origin_on_a_cell = section.getboolean('put_origin_on_a_cell')
    shelf = section.getboolean('shelf')
    hydro = section.getboolean('hyrdo')

    # Open the file, get needed dimensions
    gridfile = NetCDFFile(filename, 'r+')
    nVertLevels = len(gridfile.dimensions['nVertLevels'])
    # Get variables
    xCell = gridfile.variables['xCell']
    yCell = gridfile.variables['yCell']
    xEdge = gridfile.variables['xEdge']
    yEdge = gridfile.variables['yEdge']
    xVertex = gridfile.variables['xVertex']
    yVertex = gridfile.variables['yVertex']
    thickness = gridfile.variables['thickness']
    bedTopography = gridfile.variables['bedTopography']
    layerThicknessFractions = gridfile.variables['layerThicknessFractions']
    SMB = gridfile.variables['sfcMassBal']

    # Find center of domain
    x0 = xCell[:].min() + 0.5 * (xCell[:].max() - xCell[:].min())
    y0 = yCell[:].min() + 0.5 * (yCell[:].max() - yCell[:].min())
    # Calculate distance of each cell center from dome center
    r = ((xCell[:] - x0) ** 2 + (yCell[:] - y0) ** 2) ** 0.5

    if put_origin_on_a_cell:
        # Center the dome in the center of the cell that is closest to the
        # center of the domain.
        centerCellIndex = numpy.abs(r[:]).argmin()
        xShift = -1.0 * xCell[centerCellIndex]
        yShift = -1.0 * yCell[centerCellIndex]
        xCell[:] = xCell[:] + xShift
        yCell[:] = yCell[:] + yShift
        xEdge[:] = xEdge[:] + xShift
        yEdge[:] = yEdge[:] + yShift
        xVertex[:] = xVertex[:] + xShift
        yVertex[:] = yVertex[:] + yShift
        # Now update origin location and distance array
        x0 = 0.0
        y0 = 0.0
        r = ((xCell[:] - x0) ** 2 + (yCell[:] - y0) ** 2) ** 0.5

    # Assign variable values for dome
    # Define dome dimensions - all in meters
    r0 = 60000.0 * numpy.sqrt(0.125)
    h0 = 2000.0 * numpy.sqrt(0.125)
    # Set default value for non-dome cells
    thickness[:] = 0.0
    # Calculate the dome thickness for cells within the desired radius
    # (thickness will be NaN otherwise)
    thickness_field = thickness[0, :]
    if dome_type == 'cism':
        thickness_field[r < r0] = h0 * (1.0 - (r[r < r0] / r0) ** 2) ** 0.5
    elif dome_type == 'halfar':
        thickness_field[r < r0] = h0 * (
            1.0 - (r[r < r0] / r0) ** (4.0 / 3.0)) ** (3.0 / 7.0)
    else:
        raise ValueError('Unexpected dome_type: {}'.format(dome_type))
    thickness[0, :] = thickness_field

    # zero velocity everywhere
    # normalVelocity[:] = 0.0
    # flat bed at sea level
    bedTopography[:] = 0.0
    if shelf:
        # this line will make a small shelf:
        bedTopography[0, xCell[:] < -10000.0] = -2000.0
    # Setup layerThicknessFractions
    layerThicknessFractions[:] = 1.0 / nVertLevels

    # boundary conditions
    # Sample values to use, or comment these out for them to be 0.
    SMB[:] = 0.0
    # beta[:] = 50000.
    # units: m/yr, lapse rate of 1 m/yr with 0 at 500 m
    # SMB[:] = 2.0/1000.0 * (thickness[:] + bedTopography[:]) - 1.0
    # Convert from units of m/yr to kg/m2/s using an assumed ice density
    SMB[:] = SMB[:] * 910.0 / (3600.0 * 24.0 * 365.0)

    # lapse rate of 5 deg / km
    # Tsfc[:, 0] = -5.0/1000.0 * (thickness[0,:] + bedTopography[0,:])
    # G = 0.01
    # BMB[:] = -20.0  # units: m/yr

    if hydro:
        gridfile.variables['uReconstructX'][:] = 5.0 / (3600.0 * 24.0 * 365.0)
        gridfile.variables['basalMeltInput'][:] = 0.06 / 335000.0 * 50.0
        gridfile.variables['externalWaterInput'][:] = 0.0
        gridfile.variables['waterThickness'][:] = 0.08

    gridfile.close()

    logger.info('Successfully added dome initial conditions to: {}'.format(
        filename))
