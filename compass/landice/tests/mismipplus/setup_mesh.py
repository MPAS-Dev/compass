import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh

from compass.model import make_graph_file
from compass.step import Step


class SetupMesh(Step):
    """
    A step for creating a mesh and initial condition for MISMIP+ test cases

    Attributes
    ----------
    resolution : int
        The resolution of the test case, as defined in the configuration file
        at the time when `compass setup` is run.

    nx : int
        The number of cells in the x direction

    ny : int
        The number of cells in the y direction

    dc : int
        The distance in meters between adjacent cell centers.

    """
    def __init__(self, test_case, name, subdir, resolution):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        name : str
            the name of the test case

        subdir : str
            the subdirectory for the step.  The default is ``name``

        resolution : int
            The nominal distance [m] between horizontal grid points
        """

        super().__init__(test_case=test_case,
                         name=name,
                         subdir=subdir)

        # Files to be created as part of the this step
        for filename in ['mpas_grid.nc', 'graph.info', 'landice_grid.nc']:
            self.add_output_file(filename=filename)

        self.resolution = resolution

    def run(self):
        """
        Run this step of the test case
        """

        # hard coded parameters
        Lx = 640e3  # [m]
        Ly = 80e3   # [m]

        logger = self.logger
        config = self.config

        section = config['mesh']

        # read the resolution from the .cfg file at runtime
        resolution = section.getfloat('resolution')
        # read the gutter lenth from the .cfg file
        gutterLength = section.getfloat('gutter_length')
        # ensure that the requested `gutterLength` is valid. Otherwise set
        # the value to zero, such that the default `gutterLength` of two
        # gridcells is used.
        if (gutterLength < 2. * resolution) and (gutterLength != 0.):
            gutterLength = 0.

        # check if the resolution has been changed since the `compass setup`
        # command was run
        if self.resolution != resolution:
            raise Exception(f'Resolution was set at {self.resolution:2d}km'
                            f' when `compass setup` was called. Since then,'
                            f' the resolution in the configuration file has'
                            f' been changed to {resolution:2d}km. Changing'
                            f' resolution at runtime is not supported. Change'
                            f' the resolution value in the configuration file'
                            f' within the python module and rerun the `compass'
                            f' setup` command in order to create a mesh at a'
                            f' resolution of {resolution:2d}km')

        nx, ny, dc = calculateMeshParams(nominal_resolution=resolution,
                                         Lx=Lx,
                                         Ly=Ly,
                                         gutterLength=gutterLength)

        ds_mesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc,
                                       nonperiodic_x=True,
                                       nonperiodic_y=True)

        ds_mesh = mark_cull_cells_for_MISMIP(ds_mesh)
        ds_mesh = cull(ds_mesh, logger=logger)
        ds_mesh = convert(ds_mesh, logger=logger)

        # custom function to shift the orgin for the MISMIP+ experiments
        ds_mesh = center_trough(ds_mesh)

        write_netcdf(ds_mesh, 'mpas_grid.nc')

        # using `.get(...)`, instead of `.getint(...)` since variables need
        # to be string anyway for subprocess call
        levels = section.get('levels')
        vertMethod = section.get('vetical_layer_distribution')

        args = ['create_landice_grid_from_generic_MPAS_grid.py',
                '-i', 'mpas_grid.nc',
                '-o', 'landice_grid.nc',
                '-l', levels,
                '-v', vertMethod,
                '--diri',
                '--thermal']

        check_call(args, logger)

        make_graph_file(mesh_filename='landice_grid.nc',
                        graph_filename='graph.info')

        # create the initial conditions for the MISMIP+ spinup expr
        _setup_MISMPPlus_IC(config, logger,
                            filename='landice_grid.nc')


def calculateMeshParams(nominal_resolution,
                        Lx=640e3,
                        Ly=80e3,
                        gutterLength=0.0):
    """
    Calculate the appropriate parameters for use by `make_planar_hex_mesh`
    from the desired nominal resolution (e.g. 8e3, 4e3, 2e3, 1e3...).

    Parameters
    ----------
    nominal_resolution : int
        Desired mesh resolution in [m] without consideration of the hex meshes

    Lx : float
        Domain length in x direction [m]

    Ly : float
        Domain length in y direction [m]

    gutterLength : float
        Desired gutter length [m] on the eastern domain. Default value of 0.0
        will ensure there are two extra cell for the gutter.
    """

    def calculate_ny(nominal_resolution, Ly):
        """

        Parameters
        ----------
        nominal_resolution : int
            Desired mesh resolution in [m] without consideration of hex meshes

        Ly : float
            Domain length in y direction [m]
        """

        # Find amplitude of dual mesh (i.e. `dy`) using the nomial resolution
        nominal_dy = np.sqrt(3.) / 2. * nominal_resolution

        # find the number of y rows (`ny`) from the `nominal_dy`
        nominal_ny = Ly / nominal_dy

        # find the acutal number of y rows (`ny`) by rounding the `nominal_ny`
        # to the nearest _even_ integer. `make_planar_hex_mesh` requires that
        # `ny` be even
        ny = np.ceil(nominal_ny / 2.) * 2

        return int(ny)

    def calculate_dc(Ly, ny):
        """
        Calcuate the edge length that conforms to the desired `ny`

        Parameters
        ----------
        Ly : float
            Length [m] of y domain
        ny : int
            number of gridcells along the y-axis
        """

        # from the new ny, calculate the new dy
        dy = Ly / ny

        # from the dy, which results in an even integer number of y-rows,
        # where the cell center to cell center distance along y direction will
        # be exactly Ly, recalculate the dc.
        dc = 2. * dy / np.sqrt(3.)

        return dc

    def calculate_nx(Lx, dc, gutterLength):
        """
        Caluculate the number of x gridcell, per the required `dc` with
        considerations for the requested gutter length

        Parameters
        ----------
        Lx : float
            Length [m] of x domain

        dc : float
            Edge length [m]

        gutterLength: float
            Desired gutter length [m] on the eastern domain. A value of 0.0
            will result in two extra cell for the gutter.
        """

        if gutterLength == 0.0:
            nx = np.ceil(Lx / dc)
            # The modulo condition below ensures there is exactly one cell
            # past the the desired domain length. So, when no gutter
            # infromation is provided, add an extra column to make
            # the default gutter length 2
            nx += 1
        else:
            # ammend the domain length to account for the gutter
            Lx += gutterLength
            nx = np.ceil(Lx / dc)

        # Just rounding `nx` up to the nearest `int` doesn't gurantee that the
        # domain will fully span [0, Lx]. If dc/2 (i.e. `dx`) is less than the
        # than the remainder of `Lx / dc` even the rounded up `nx` will fall
        # short of the desired domain length. If that's the case add an extra
        # x cell so the domain is inclusive of the calving front.
        if Lx % dc > dc / 2.:
            nx += 1

        return int(nx)

    ny = calculate_ny(nominal_resolution, Ly)
    dc = calculate_dc(Ly, ny)
    nx = calculate_nx(Lx, dc, gutterLength)

    # add two to `ny` accomodate MISMIP+ specific culling requirments.
    # This ensures, after the culling, that the cell center too cell center
    # distance in the y-direction is exactly equal to `Ly`. This must be done
    # after the other parameters (i.e. `dc` and `nx`) are calculated or else
    # those calculations will be thrown off.
    return nx, ny + 2, dc


def mark_cull_cells_for_MISMIP(ds_mesh):
    """

    Parameters
    ----------
    ds_mesh : xarray.Dataset
    """

    # get the edge length [m] from the attributes
    dc = ds_mesh.dc
    # calculate the amplitude (i.e. `dy`) [m] of the dual mesh
    dy = np.sqrt(3.) / 2. * dc
    # Get the y position of the top row
    yMax = ds_mesh.yCell.max()

    # find the first interior row along the top of the domain
    mask = np.isclose(ds_mesh.yCell, yMax - dy, rtol=0.02)

    # add first interior row along northern boudnary to the cells to be culled
    ds_mesh['cullCell'] = xr.where(mask, 1, ds_mesh.cullCell)

    return ds_mesh


def center_trough(ds_mesh):
    """
    Shift the origin so that the bed trough is centered about the Y-axis and
    the X-axis is shifted all the way to the left.

    Parameters
    ----------
    ds_mesh : xarray.Datatset
        The mesh to be shifted
    """

    # Get the center of y-axis (i.e. half distance)
    yCenter = (ds_mesh.yCell.max() + ds_mesh.yCell.min()) / 2.

    # Shift x-axis so that the x-origin is all the way to the left
    xShift = -1.0 * ds_mesh.xCell.min()
    # Shift y-axis so that it's centered about the MISMIP+ bed trough
    yShift = 4.e4 - yCenter

    # Shift all spatial points by calculated shift
    for loc in ['Cell', 'Edge', 'Vertex']:
        ds_mesh[f'x{loc}'] = ds_mesh[f'x{loc}'] + xShift
        ds_mesh[f'y{loc}'] = ds_mesh[f'y{loc}'] + yShift

    ##########################################################################
    # WHL :
    #   Need to adjust geometry along top and bottom boundaries to get flux
    #   correct there. Essentially, we only want to model the interior half of
    #   those cells. Adding this here because we only want to do this if it
    #   hasn't been done before. This method is assuming a periodic_hex mesh!
    ##########################################################################

    # Boolean mask for indices which correspond to the N/S boundary of mesh
    # `np.isclose` is needed when comparing floats to avoid roundoff errors
    mask = (np.isclose(ds_mesh.yCell, ds_mesh.yCell.min(), rtol=0.01) |
            np.isclose(ds_mesh.yCell, ds_mesh.yCell.max(), rtol=0.01))

    # Reduce the cell areas by half along the N/S boundary
    ds_mesh['areaCell'] = xr.where(mask,
                                   ds_mesh.areaCell * 0.5,
                                   ds_mesh.areaCell)

    # `mesh.dcEdge` is a vector. So, ensure that all values equal before we
    # arbitrarily select a value from the array to use in the `de` calculation
    if ds_mesh.dcEdge.all():
        dc = float(ds_mesh.dcEdge[0])

    # get the distance between edges
    de = 0.5 * dc * np.sin(np.pi / 3)
    # find the min and max (i.e. N/S boundary) edges
    yMin = ds_mesh.yEdge.min()
    yMax = ds_mesh.yEdge.max()

    # Boolean mask for edge indices on the N/S boundary of the mesh
    mask = (np.isclose(ds_mesh.yEdge, yMin, rtol=0.01) |
            np.isclose(ds_mesh.yEdge, yMax, rtol=0.01))
    # WHL: zero out the edges on the boundary
    #      (not necessary because velocity will also be zero)
    ds_mesh['dvEdge'] = xr.where(mask, 0.0, ds_mesh.dvEdge)

    # Boolean mask for the indexed of edges N/S of  boundary cell centers,
    # using a 2% relative threshold to account for accumulated roundoff
    # from min calculation
    mask = (np.isclose(ds_mesh.yEdge, yMin + de, rtol=0.02) |
            np.isclose(ds_mesh.yEdge, yMax - de, rtol=0.02))
    # cut length in half for edges between boundary cells
    ds_mesh['dvEdge'] = xr.where(mask, ds_mesh.dvEdge * 0.5, ds_mesh.dvEdge)

    return ds_mesh


def _mismipplus_bed(x, y):
    """
    Equations 1-4 from Asay-Davis et al. (2016).

    Parameters
    ----------
    x : xarray.DataArray
        x cell coordinates

    y : xarray.DataArray
        y cell coordinates
    """
    x = x / 1.e3      # m to km
    y = y / 1.e3      # m to km
    B0 = -150.        # m
    B2 = -728.8       # m
    B4 = 343.91       # m
    B6 = -50.57       # m
    x_bar = 300.      # km
    x_tilde = x / x_bar
    dc = 500.         # m
    fc = 4.           # km
    wc = 24.          # km
    Ly = 80.          # km
    Bmax = -720.      # m

    B_x = B0 + B2 * x_tilde**2 + B4 * x_tilde**4 + B6 * x_tilde**6
    B_y = dc / (1 + np.exp(-2 * (y - Ly / 2 - wc) / fc)) +\
        dc / (1 + np.exp(2 * (y - Ly / 2 + wc) / fc))

    Bsum = B_x + B_y

    z_b = np.maximum(Bsum, Bmax)  # Eqn 1. (Asay-Davis et al. 2016)

    return z_b


def _setup_MISMPPlus_IC(config, logger, filename):
    """
    Add the inital condition for the MISMIP+ spinup to the given MPAS mesh file

    Parameters
    ----------
    config : comass.config.CompassConfigParser
        Configuration options for this test case

    logger : logging.Logger
        A logger for output from the step

    filename : str
        NetCDF file to place the MISMIP+ ICs in

    """

    # Hard code some parameters from Table 1. of Asay-Davis et al. 2016
    accum = 0.3       # m^{-1}
    C = 3.160e6       # Pa m^{-1/3} s^{1/3}
    rhoi = 918        # kg m^{-3}
    spy = 31556926    # s a^{-1}
    xcalve = 640.e3   # m

    # Read parameters from the .cfg file
    section = config['mesh']
    init_thickness = section.getfloat('init_thickness')

    # open the file
    src = xr.open_dataset(filename)

    # Use `.loc[:]` for setting the initial conditions since we are setting
    # the fields with scalar values and/or fields with a reduced number of
    # dimensions. This ensures values are properly broadcast and aligned
    # against the field's coordinates.

    # Set the bedTopography
    src['bedTopography'].loc[:] = _mismipplus_bed(src.xCell, src.yCell)

    # Set the ice thickness
    src['thickness'].loc[:] = xr.where(src.xCell < xcalve, init_thickness, 0.)

    # Convert SMB from m/yr to kg/m2/s
    accum *= rhoi / spy
    # Set the surface mass balance
    src['sfcMassBal'].loc[:] = accum

    # create the calving mask
    mask = src.xCell > xcalve
    # create the calvingMask data array and add Time dimension along axis 0
    calvingMask = xr.where(mask, 1, 0).expand_dims({"Time": 1}, axis=0)
    # assign data array to dataset and ensure it's a 32 bit int field
    src['calvingMask'] = calvingMask.astype('int32')

    # Boolean masks for indices which correspond to the N/S boundary of mesh
    mask = (np.isclose(src.yCell, src.yCell.min(), rtol=0.01) |
            np.isclose(src.yCell, src.yCell.max(), rtol=0.02))
    # NOTE: np.isclose returns a np.array. Due to the bug in xarray (<=2023.8)
    #       mask variable needs to converted to an xarray object in order for
    #       the `.variable` attribute to exist (which is needed to fix the
    #       bug in the broadcasting/alignment when `.loc[:]` is used.)
    mask = xr.DataArray(mask, dims='nCells')
    # Set free slip boundary conditions for velocity along N/S boundary
    # NOTE: `.variable` is needed so that coordinates are properly broadcast
    #        due to a bug in xarray, which was resolved as of 2023.9.0
    src['dirichletVelocityMask'].loc[:] = xr.where(mask, 1, 0).variable

    # Skipping WHL step of setting the initial velocities to zero
    # since they are already zero.

    # convert to MPAS units
    C /= spy**(1.0 / 3.0)

    # Set the effectivePressure using a Weertman power law
    src['effectivePressure'].loc[:] = C

    # Write the dataset to disk
    write_netcdf(src, filename)

    print(f'Successfully added MISMIP+ initial conditions to: {filename}')
