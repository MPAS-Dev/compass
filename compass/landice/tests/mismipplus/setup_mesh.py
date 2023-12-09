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
    def __init__(self, test_case, resolution):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : int
            The distance between horizontal grid points
        """
        # Defined resolutions, the value of `resolution` in the configuration
        # file needs to a key in the dictionary, or added manually.
        resolution_params = {'8km': {'nx': 73, 'ny': 14, 'dc': 9237.604307},
                             '4km': {'nx': 144, 'ny': 24, 'dc': 4618.802154},
                             '2km': {'nx': 284, 'ny': 44, 'dc': 2309.401077},
                             '1km': {'nx': 566, 'ny': 84, 'dc': 1154.700538}}

        # With the defined resolution, make the associated key for the param
        # dictionary (necessary b/c unstructured hex meshes)
        resolution_key = f'{resolution}km'

        # Ensure the resolution passed in the configuration file is defined
        # within the resolution parameter dictionary.
        assert_error_msg = (f"Resolution of {resolution} km is not defined in"
                            f" the `resolution_params` dictionary. Valid"
                            f" options  are {list(resolution_params.keys())}."
                            f" Either choose one of the existing resolutions"
                            f" or add the appropriate parameter values (i.e."
                            f" nx, ny, dx) to the dictionary for"
                            f" the resolution you want.")

        assert resolution_key in resolution_params, assert_error_msg

        super().__init__(test_case=test_case,
                         name=f'{resolution_key}_mesh_gen',
                         subdir=f"{resolution_key}/mesh_gen")

        # Files to be created as part of the this step
        for filename in ['mpas_grid.nc', 'graph.info', 'landice_grid.nc']:
            self.add_output_file(filename=filename)

        # unpack the mesh parameters for the given resolution as set as
        # attributes
        self.nx, self.ny, self.dc = resolution_params[resolution_key].values()

        self.resolution = resolution

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config

        section = config['mesh']

        # read the resolution from the .cfg file at runtime
        resolution = section.getint('resolution')

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

        ds_mesh = make_planar_hex_mesh(nx=self.nx, ny=self.ny, dc=self.dc,
                                       nonperiodic_x=True,
                                       nonperiodic_y=True)

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


def center_trough(ds_mesh):
    """
    Shift the origin so that the bed trough is centered about the Y-axis and
    the X-axis is shifted all the way to the left.

    Parameters
    ----------
    ds_mesh : xr.core.dataset.Datatset
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

    # WHL :
    #   Need to adjust geometry along top and bottom boundaries to get flux
    #   correct there. Essentially, we only want to model the interior half of
    #   those cells. Adding this here because we only want to do this if it
    #   hasn't been done before. This method is assuming a periodic_hex mesh!

    # Reduce the cell areas by half along the N/S boundary

    # Boolean mask for indices which correspond to the N/S boundary of mesh
    mask = ((ds_mesh.yCell == ds_mesh.yCell.min()) |
            (ds_mesh.yCell == ds_mesh.yCell.max()))

    ds_mesh['areaCell'] = xr.where(mask,
                                   ds_mesh.areaCell * 0.5,
                                   ds_mesh.areaCell)

    # Needed of reducing the edge lengths by half along the N/S boundary.
    # Values returned by `np.unique` will be sorted.
    unique_ys_edge = np.unique(ds_mesh.yEdge.data)

    # Boolean mask for edge indices on the N/S boundary of the mesh
    mask = ((ds_mesh.yEdge == unique_ys_edge[0]) |
            (ds_mesh.yEdge == unique_ys_edge[-1]))
    # WHL: zero out the edges on the boundary
    #      (not necessary because velocity will also be zero)
    ds_mesh['dvEdge'] = xr.where(mask, 0.0, ds_mesh.dvEdge)

    # Boolean mask for the indexed of edges between N/S boundary cells
    mask = ((ds_mesh.yEdge == unique_ys_edge[1]) |
            (ds_mesh.yEdge == unique_ys_edge[-2]))
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

    # Use `.loc[:]` for setting the initial conditions  since we are setting
    # the fields with scalar values. This ensures values are properly broadcast
    # against the field's coordinates

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
    mask = (src.yCell == src.yCell.min()) | (src.yCell == src.yCell.max())
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
