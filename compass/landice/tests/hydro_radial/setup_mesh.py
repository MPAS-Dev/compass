import numpy as np
from netCDF4 import Dataset as NetCDFFile
from importlib.resources import path

from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.logging import check_call

from compass.model import make_graph_file
from compass.step import Step


class SetupMesh(Step):
    """
    A step for creating a mesh and initial condition for dome test cases

    Attributes
    ----------
    initial_condition : {'zero', 'exact'}
        The type of initial condition to set up.  'zero' means nearly zero ice
        thickness.  'exact' uses a precomputed near exact solution from a file.
    """
    def __init__(self, test_case, initial_condition):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        initial_condition : {'zero', 'exact'}
            The type of initial condition to set up.  'zero' means nearly zero
            ice thickness.  'exact' uses a precomputed near exact solution from
            a file.
        """
        super().__init__(test_case=test_case, name='setup_mesh')

        self.initial_condition = initial_condition

        if initial_condition == 'exact':
            filename = 'near_exact_solution_r_P_W.txt'
            with path('compass.landice.tests.hydro_radial', filename) as target:
                self.add_input_file(filename=filename, target=str(target))
        elif initial_condition != 'zero':
            raise ValueError("Unknown initial condition type specified "
                             "{}.".format(initial_condition))

        self.add_output_file(filename='graph.info')
        self.add_output_file(filename='landice_grid.nc')

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
        """
        initial_condition = self.initial_condition
        logger = self.logger
        section = self.config['hydro_radial']

        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=True,
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
                '--hydro',
                '--diri']

        check_call(args, logger)

        make_graph_file(mesh_filename='landice_grid.nc',
                        graph_filename='graph.info')

        _setup_hydro_radial_initial_conditions(
            logger, filename='landice_grid.nc',
            initial_condition=initial_condition)


def _setup_hydro_radial_initial_conditions(logger, filename,
                                           initial_condition):
    """
    Add the initial condition to the given MPAS mesh file

    Parameters
    ----------
    logger : logging.Logger
        A logger for output from the step

    filename : str
        file to setup hydro_radial

    initial_condition : {'zero', 'exact'}
        the type of initial condition
    """
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

    # Find center of domain
    x0 = xCell[:].min() + 0.5 * (xCell[:].max() - xCell[:].min())
    y0 = yCell[:].min() + 0.5 * (yCell[:].max() - yCell[:].min())
    # Calculate distance of each cell center from dome center
    r = ((xCell[:] - x0)**2 + (yCell[:] - y0)**2)**0.5

    # Center the dome in the center of the cell that is closest to the center
    # of the domain.
    #   NOTE: for some meshes, maybe we don't want to do this - could add
    #   command-line argument controlling this later.
    putOriginOnACell = True
    if putOriginOnACell:
        centerCellIndex = np.abs(r[:]).argmin()
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
        r = ((xCell[:] - x0)**2 + (yCell[:] - y0)**2)**0.5

    # center thickness (m)
    h0 = 500.0
    # sliding velocity at margin (m/s)
    v0 = 100.0 / (3600.0 * 24.0 * 365.0)
    # ideal ice cap radius (m)
    R0 = 25.0e3
    # onset of sliding (m)
    R1 = 5.0e3
    # actual margin location (m)
    L = 0.9 * R0

    thickness[0, r < R0] = h0 * (1.0 - (r[r < R0] / R0)**2)
    thickness[0, r > L] = 0.0

    # flat bed
    bedTopography[:] = 0.0

    # Setup layerThicknessFractions
    layerThicknessFractions[:] = 1.0 / nVertLevels

    # melt
    gridfile.variables['basalMeltInput'][:] = 0.0
    # 20 cm/yr as SI mass rate
    gridfile.variables['basalMeltInput'][:] = \
        0.2 / (365.0 * 24.0 * 3600.0) * 1000.0
    # Use this line to only add a source term to the center cell - useful for
    # debugging divergence

    # value from ramp
    # gridfile.variables['basalMeltInput'][0,r==0.0] = 4.0e-10 * 1000.0 *100

    # velocity
    gridfile.variables['uReconstructX'][:] = 0.0
    velo = v0 * (r - R1)**5 / (L - R1)**5
    velo[r < R1] = 0.0
    gridfile.variables['uReconstructX'][0, :, -1] = velo
    gridfile.variables['uReconstructX'][0, thickness[0, :] == 0.0, :] = 0.0

    if initial_condition == 'zero':
        logger.info("Using 'zero' option for initial condition.")
        # set some small initial value to keep adaptive time stepper from
        # taking a huge time step initially
        gridfile.variables['waterThickness'][0, :] = 0.01
        gridfile.variables['waterPressure'][0, :] = 0.0
    elif initial_condition == 'exact':
        logger.info("Using 'exact' option for initial condition.")
        # IC on thickness
        # import exact solution
        fnameSoln = 'near_exact_solution_r_P_W.txt'
        soln = np.loadtxt(fnameSoln, delimiter=',')
        rsoln = soln[:, 0]
        Psoln = soln[:, 1]
        Wsoln = soln[:, 2]

        Wmpas = np.interp(r, rsoln, Wsoln)  # apply exact solution
        Wmpas[np.isnan(Wmpas)] = 0.0
        gridfile.variables['waterThickness'][0, :] = Wmpas

        # IC on water pressure
        # apply exact solution
        Pmpas = np.interp(r, rsoln, Psoln)
        Pmpas[np.isnan(Pmpas)] = 0.0
        gridfile.variables['waterPressure'][0, :] = Pmpas
    else:
        raise ValueError("Unknown initial condition type specified "
                         "{}.".format(initial_condition))

    gridfile.close()

    logger.info('Successfully added hydro_radial initial conditions to: '
                '{}'.format(filename))
