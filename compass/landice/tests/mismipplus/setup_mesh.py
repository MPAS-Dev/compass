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
    Experimental protocol described in:
    Asay-Davis, et al. 2016. "Experimental Design for Three Interrelated
    Marine Ice Sheet and Ocean Model Intercomparison Projects:
    MISMIP v.3 (MISMIP+), ISOMIP v.2 (ISOMIP+) and MISOMIP v.1 (MISOMIP1)."
    Geoscientific Model Development 9 (7): 2471–97.
    https://doi.org/10.5194/gmd-9-2471-2016.

    Attributes
    ----------
    resolution : float
        The resolution of the test case, as defined in the configuration file
        at the time when `compass setup` is run.

    nx : int
        The number of cells in the x direction

    ny : int
        The number of cells in the y direction

    dc : int
        The distance in meters between adjacent cell centers.

    """
    def __init__(self, test_case, name, subdir=None):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        name : str
            the name of the test case

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``
        """

        super().__init__(test_case=test_case,
                         name=name,
                         subdir=subdir)

        # Files to be created as part of the this step
        for filename in ['mpas_grid.nc', 'graph.info', 'landice_grid.nc']:
            self.add_output_file(filename=filename)

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
        gutter_length = section.getfloat('gutter_length')
        # ensure that the requested `gutter_length` is valid. Otherwise set
        # the value to zero, such that the default `gutter_length` of two
        # gridcells is used.
        if (gutter_length < 2. * resolution) and (gutter_length != 0.):
            gutter_length = 0.

        # check if the resolution has been changed since the `compass setup`
        # command was run
        if self.resolution != resolution:
            raise ValueError(f'Resolution was set at {self.resolution:4.0f}m'
                             f' when `compass setup` was called. Since then,'
                             f' the resolution in the configuration file has'
                             f' been changed to {resolution:4.0f}m. Changing'
                             f' resolution at runtime is not supported. Change'
                             f' the resolution value in the configuration file'
                             f' within the python module and rerun the'
                             f' `compass setup` command in order to create'
                             f' a mesh at a resolution of {resolution:4.0f}m')

        nx, ny, dc = calculate_mesh_params(nominal_resolution=resolution,
                                           Lx=Lx,
                                           Ly=Ly,
                                           gutter_length=gutter_length)

        ds_mesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc,
                                       nonperiodic_x=True,
                                       nonperiodic_y=True)

        ds_mesh = mark_cull_cells_for_MISMIP(ds_mesh)
        ds_mesh = cull(ds_mesh, logger=logger)
        ds_mesh = convert(ds_mesh, logger=logger)

        # custom function to shift the origin for the MISMIP+ experiments
        ds_mesh = center_trough(ds_mesh)

        write_netcdf(ds_mesh, 'mpas_grid.nc')

        # using `.get(...)`, instead of `.getint(...)` since variables need
        # to be string anyway for subprocess call
        levels = section.get('levels')
        vertMethod = section.get('vetical_layer_distribution')

        args = ['create_landice_grid_from_generic_mpas_grid',
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


def calculate_mesh_params(nominal_resolution,
                          Lx=640e3,
                          Ly=80e3,
                          gutter_length=0.0):
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

    gutter_length : float
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

        # Find amplitude of dual mesh (i.e. `dy`) using the nominal resolution
        nominal_dy = np.sqrt(3.) / 2. * nominal_resolution

        # find the number of y rows (`ny`) from the `nominal_dy`
        nominal_ny = Ly / nominal_dy

        # find the actual number of y rows (`ny`) by rounding the `nominal_ny`
        # to the nearest _even_ integer. `make_planar_hex_mesh` requires that
        # `ny` be even
        ny = np.ceil(nominal_ny / 2.) * 2

        return int(ny)

    def calculate_dc(Ly, ny):
        """
        Calculate the cell spacing that conforms to the desired `ny`

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

    def calculate_nx(Lx, dc, gutter_length):
        """
        Caluculate the number of x gridcell, per the required `dc` with
        considerations for the requested gutter length

        Parameters
        ----------
        Lx : float
            Length [m] of x domain

        dc : float
            Edge length [m]

        gutter_length: float
            Desired gutter length [m] on the eastern domain. A value of 0.0
            will result in two extra cell for the gutter.
        """

        if gutter_length == 0.0:
            nx = np.ceil(Lx / dc)
            # The modulo condition below ensures there is exactly one cell
            # past the the desired domain length. So, when no gutter
            # information is provided, add an extra column to make
            # the default gutter length 2
            nx += 1
        else:
            # amend the domain length to account for the gutter
            Lx += gutter_length
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
    nx = calculate_nx(Lx, dc, gutter_length)

    # add two to `ny` to accomodate MISMIP+ specific culling requirments.
    # This ensures, after the culling, that the cell center to cell center
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
    y_max = ds_mesh.yCell.max()

    # set the absolute tolerance to +/- 5% of `dy`
    atol = dy * 0.05
    # find the first interior row along the top of the domain
    mask = np.isclose(ds_mesh.yCell, y_max - dy, atol=atol, rtol=0)

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
    y_center = (ds_mesh.yCell.max() + ds_mesh.yCell.min()) / 2.

    # Shift x-axis so that the x-origin is all the way to the left
    x_shift = -1.0 * ds_mesh.xCell.min()
    # Shift y-axis so that it's centered about the MISMIP+ bed trough
    y_shift = 4.e4 - y_center

    # Shift all spatial points by calculated shift
    for loc in ['Cell', 'Edge', 'Vertex']:
        ds_mesh[f'x{loc}'] = ds_mesh[f'x{loc}'] + x_shift
        ds_mesh[f'y{loc}'] = ds_mesh[f'y{loc}'] + y_shift

    # `mesh.dcEdge` is a vector. So, ensure that all values equal before we
    # arbitrarily select a value from the array to use in the `de` calculation
    if ds_mesh.dcEdge.all():
        dc = float(ds_mesh.dcEdge[0])

    # calculate the amplitude (i.e. `dy`) [m] of the dual mesh
    dy = np.sqrt(3.) / 2. * dc
    # Get the y position of the top/bottom row
    y_max = ds_mesh.yCell.max()
    y_min = ds_mesh.yCell.min()

    ##########################################################################
    # WHL :
    #   Need to adjust geometry along top and bottom boundaries to get flux
    #   correct there. Essentially, we only want to model the interior half of
    #   those cells. Adding this here because we only want to do this if it
    #   hasn't been done before. This method is assuming a periodic_hex mesh!
    ##########################################################################

    # Boolean mask for indices which correspond to the N/S boundary of mesh
    # `np.isclose` is needed when comparing floats to avoid roundoff errors
    mask = (np.isclose(ds_mesh.yCell, y_min, atol=dy * 0.05, rtol=0) |
            np.isclose(ds_mesh.yCell, y_max, atol=dy * 0.05, rtol=0))

    # Reduce the cell areas by half along the N/S boundary
    ds_mesh['areaCell'] = xr.where(mask,
                                   ds_mesh.areaCell * 0.5,
                                   ds_mesh.areaCell)

    # get the distance between edges. Since all meshes are generated with the
    # `make_planar_hex_mesh` function, all triangles (in the dual mesh) will
    # be equilateral, which makes our use of `3` in the denominator below
    # a valid assumption.
    de = 0.5 * dc * np.sin(np.pi / 3)
    # find the min and max (i.e. N/S boundary) edges
    y_min = ds_mesh.yEdge.min()
    y_max = ds_mesh.yEdge.max()

    # Boolean mask for edge indices on the N/S boundary of the mesh
    mask = (np.isclose(ds_mesh.yEdge, y_min, atol=de * 0.05, rtol=0) |
            np.isclose(ds_mesh.yEdge, y_max, atol=de * 0.05, rtol=0))
    # WHL: zero out the edges on the boundary
    #      (not necessary because velocity will also be zero)
    ds_mesh['dvEdge'] = xr.where(mask, 0.0, ds_mesh.dvEdge)

    # Boolean mask for the indexed of edges N/S of  boundary cell centers,
    # using an absolute threshold of 5% of the edge distance
    mask = (np.isclose(ds_mesh.yEdge, y_min + de, atol=de * 0.05, rtol=0) |
            np.isclose(ds_mesh.yEdge, y_max - de, atol=de * 0.05, rtol=0))
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
    config : compass.config.CompassConfigParser
        Configuration options for this test case

    logger : logging.Logger
        A logger for output from the step

    filename : str
        NetCDF file to place the MISMIP+ ICs in

    """

    # Hard code some parameters from Table 1. of Asay-Davis et al. 2016
    accum = 0.3       # m^{-1}
    C = 3.160e6       # Pa m^{-1/3} s^{1/3}
    spy = 31556926    # s a^{-1}
    xcalve = 640.e3   # m

    # Read parameters from the .cfg file
    section = config['mesh']
    rhoi = section.getfloat('ice_density')               # kg m^{-3}
    init_thickness = section.getfloat('init_thickness')  # m

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

    # `src.dcEdge` is a vector. So, ensure that all values equal before we
    # arbitrarily select a value from the array to use in the `de` calculation
    if src.dcEdge.all():
        dc = float(src.dcEdge[0])

    # calculate the amplitude (i.e. `dy`) [m] of the dual mesh
    dy = np.sqrt(3.) / 2. * dc
    # Get the y position of the top/bottom row
    y_max = src.yCell.max()
    y_min = src.yCell.min()

    # Boolean masks for indices which correspond to the N/S boundary of mesh
    mask = (np.isclose(src.yCell, y_min, atol=dy * 0.05, rtol=0) |
            np.isclose(src.yCell, y_max, atol=dy * 0.05, rtol=0))
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

    # convert to MPAS units (m yr^{-1})^(1-1/n)
    C /= spy**(1.0 / 3.0)

    # Set the basal traction parameter for Weetman sliding law
    src['muFriction'].loc[:] = C

    # Set the effectivePressure to constant and uniform value
    src['effectivePressure'].loc[:] = 1.0

    # Write the dataset to disk
    write_netcdf(src, filename)

    print(f'Successfully added MISMIP+ initial conditions to: {filename}')
