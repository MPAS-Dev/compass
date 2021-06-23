import numpy
from netCDF4 import Dataset as NetCDFFile

from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.logging import check_call

from compass.model import make_graph_file
from compass.step import Step


class SetupMesh(Step):
    """
    A step for creating a mesh and initial condition for circular_shelf
    test cases

    Attributes
    ----------
    mesh_type : str
        The resolution or mesh type of the test case
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
        self.mesh_type = test_case.mesh_type

        self.add_output_file(filename='graph.info')
        self.add_output_file(filename='landice_grid.nc')

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        section = config['circular_shelf']

        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc,
                                      nonperiodic_x=True,
                                      nonperiodic_y=True)

        write_netcdf(dsMesh, 'grid.nc')

        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, logger=logger)
        write_netcdf(dsMesh, 'mpas_grid.nc')

        levels = section.get('levels')
        args = ['create_landice_grid_from_generic_MPAS_grid.py',
                '-i', 'mpas_grid.nc',
                '-o', 'landice_grid.nc',
                '-l', levels,
                '--diri']

        check_call(args, logger)

        make_graph_file(mesh_filename='landice_grid.nc',
                        graph_filename='graph.info')

        _setup_circular_shelf_initial_conditions(config, logger,
                                                 filename='landice_grid.nc')


def _setup_circular_shelf_initial_conditions(config, logger, filename):
    """
    Add the initial condition to the given MPAS mesh file

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step

    filename : str
        file to setup circular_shelf
    """
    section = config['circular_shelf']
    use_mu = section.getboolean('use_mu')
    use_7cells = section.getboolean('use_7cells')

    # Open the file, get needed dimensions
    gridfile = NetCDFFile(filename, 'r+')
    nVertLevels = len(gridfile.dimensions['nVertLevels'])
    if nVertLevels != 5:
        logger.info('nVertLevels in the supplied file was {}.  '
                    'This test case is typically run with 5 levels.'
                    .format(nVertLevels))
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
    cellsOnCell = gridfile.variables['cellsOnCell']
    # Get b.c. variables
    SMB = gridfile.variables['sfcMassBal']

    # Center the circular_shelf in the center of the cell that is closest to
    # the center of the domain.
    # Only do this if it appears this has not already been done:
    if xVertex[:].min() >= 0.0:
        logger.info("Shifting x/y coordinates to center domain at 0,0.")
        # Find center of domain
        x0 = xCell[:].min() + 0.5 * (xCell[:].max() - xCell[:].min())
        y0 = yCell[:].min() + 0.5 * (yCell[:].max() - yCell[:].min())
        # Calculate distance of each cell center from circular_shelf center
        r = ((xCell[:] - x0)**2 + (yCell[:] - y0)**2)**0.5
        centerCellIndex = numpy.abs(r[:]).argmin()
        xShift = -1.0 * xCell[centerCellIndex]
        yShift = -1.0 * yCell[centerCellIndex]
        xCell[:] = xCell[:] + xShift
        yCell[:] = yCell[:] + yShift
        xEdge[:] = xEdge[:] + xShift
        yEdge[:] = yEdge[:] + yShift
        xVertex[:] = xVertex[:] + xShift
        yVertex[:] = yVertex[:] + yShift
    # Now update our local values of the origin location and distance array
    # (or assume these are correct because this grid has previously been
    # shifted)
    x0 = 0.0
    y0 = 0.0
    r = ((xCell[:] - x0)**2 + (yCell[:] - y0)**2)**0.5
    centerCellIndex = numpy.abs(r[:]).argmin()

    # Make a circular ice mass
    # Define circular_shelf dimensions - all in meters
    r0 = 21000.0
    thickness[:] = 0.0  # initialize to 0.0
    # Calculate the circular_shelf thickness for cells within the desired
    # radius (thickness will be NaN otherwise)
    thickness_field = thickness[0, :]
    thickness_field[r < r0] = 1000.0
    thickness[0, :] = thickness_field

    # flat bed at -2000 m everywhere with a single grounded point
    bedTopography[:] = -2000.0
    bedTopography[0, centerCellIndex] = -880.0
    if use_7cells:
        logger.info('Making the grounded portion of the domain cover 7 cells '
                    '- the center cell and its 6 neighbors.')
        bedTopography[0, cellsOnCell[centerCellIndex, :] - 1] = -880.0
    else:
        logger.info('Making the grounded portion of the domain cover 1 cell'
                    '- the center cell.')

    if use_mu:
        logger.info('Setting no-slip on the grounded portion of the domain'
                    'by setting a high mu field there.')
        mu = gridfile.variables['muFriction']
        # mu is 0 everywhere except a high value in the grounded cell
        mu[:] = 0.0
        mu[centerCellIndex] = 1.0e8
        if use_7cells:
            mu[cellsOnCell[centerCellIndex, :] - 1] = 1.0e8
    else:  # use Dirichlet b.c.
        logger.info('Setting no-slip on the grounded portion of the domain'
                    'by setting no-slip Dirichlet velocity boundary'
                    'conditions there.')
        dirMask = gridfile.variables['dirichletVelocityMask']
        uvel = gridfile.variables['uReconstructX']
        vvel = gridfile.variables['uReconstructY']
        dirMask[:] = 0
        uvel[:] = 0.0
        vvel[:] = 0.0
        # Apply mask to basal level of grounded portion only
        dirMask[:, centerCellIndex, -1] = 1
        if use_7cells:
            dirMask[:, cellsOnCell[centerCellIndex, :] - 1, -1] = 1

    # Setup layerThicknessFractions
    layerThicknessFractions[:] = 1.0 / nVertLevels
    # boundary conditions
    SMB[:] = 0.0  # m/yr
    # Convert from units of m/yr to kg/m2/s using an assumed ice density
    SMB[:] = SMB[:] * 910.0 / (3600.0 * 24.0 * 365.0)

    gridfile.close()

    logger.info('Successfully added circular shelf initial conditions to: {}'
                .format(filename))
