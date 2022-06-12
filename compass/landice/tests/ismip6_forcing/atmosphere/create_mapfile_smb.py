import os
import numpy as np
import shutil
import subprocess
import netCDF4
import xarray as xr
from mpas_tools.scrip.from_mpas import scrip_from_mpas
from mpas_tools.logging import check_call
from pyremap.descriptor import interp_extrap_corners_2d


# function that creates a mapping file from ismip6 grid to mali mesh
def build_mapping_file(config, cores, logger, ismip6_grid_file,
                       mapping_file, mali_mesh_file=None, method_remap=None):
    """
    Build a mapping file if it does not exist.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for a ismip6 forcing test case
    cores : int
        the number of cores for the ESMF_RegridWeightGen
    logger : logging.Logger
        A logger for output from the step
    ismip6_grid_file : str
        ismip6 grid file
    mapping_file : str
        weights for interpolation from ismip6_grid_file to mali_mesh_file
    mali_mesh_file : str, optional
        The MALI mesh file is used if mapping file does not exist
    method_remap : str, optional
        Remapping method used in building a mapping file
    """

    if os.path.exists(mapping_file):
        print("Mapping file exists. Not building a new one.")
        return

    # create the scrip files if mapping file does not exist
    print("Mapping file does not exist. Building one based on the "
          "input/ouptut meshes")
    print("Creating temporary scrip files for source and destination grids...")

    if mali_mesh_file is None:
        raise ValueError("Mapping file does not exist. To build one, Mali "
                         "mesh file with '-f' should be provided. "
                         "Type --help for info")

    # name temporary scrip files that will be used to build mapping file
    source_grid_scripfile = "temp_source_scrip.nc"
    mali_scripfile = "temp_mali_scrip.nc"
    # this is the projection of ismip6 data for Antarctica
    ismip6_projection = "ais-bedmap2"

    # create a scripfile for the atmosphere forcing data
    create_atm_scrip(ismip6_grid_file, source_grid_scripfile)

    # create a MALI mesh scripfile
    # make sure the mali mesh file uses the longitude convention of [0 2pi]
    # make changes on a duplicated file to avoid making changes to the
    # original mesh file

    mali_mesh_copy = f"{mali_mesh_file}_copy"
    shutil.copy(mali_mesh_file, f"{mali_mesh_file}_copy")

    args = ["set_lat_lon_fields_in_planar_grid.py",
            "--file", mali_mesh_copy,
            "--proj", ismip6_projection]

    subprocess.check_call(args)

    # create a MALI mesh scripfile if mapping file does not exist
    scrip_from_mpas(mali_mesh_copy, mali_scripfile)

    # create a mapping file using ESMF weight gen
    print(f"Creating a mapping file. Mapping method used: {method_remap}")

    if method_remap is None:
        raise ValueError("Desired remapping option should be provided with "
                         "--method. Available options are 'bilinear',"
                         "'neareststod', 'conserve'.")

    parallel_executable = config.get('parallel', 'parallel_executable')
    # split the parallel executable into constituents in case it includes flags
    # args = parallel_executable.split(' ')
    # args.extend(["-n", f"{cores}",
    #              "ESMF_RegridWeightGen",
    #              "-s", source_grid_scripfile,
    #              "-d", mali_scripfile,
    #              "-w", mapping_file,
    #              "-m", method_remap,
    #              "-i", "-64bit_offset",
    #              "--dst_regional", "--src_regional"])
    #
    # check_call(args, logger)

    args = ["ESMF_RegridWeightGen",
            "-s", source_grid_scripfile,
            "-d", mali_scripfile,
            "-w", mapping_file,
            "-m", method_remap,
            "-i", "-64bit_offset",
            "--dst_regional", "--src_regional"]

    subprocess.check_call(args)

    # remove the temporary scripfiles once the mapping file is generated
    print("Removing the temporary mesh and scripfiles...")
    # os.remove(source_grid_scripfile)
    os.remove(mali_scripfile)
    os.remove(mali_mesh_copy)


def create_atm_scrip(source_grid_file, source_grid_scripfile):
    """
    create a scripfile for the ismip6 atmospheric forcing data.
    Note: the atmospheric forcing data do not have 'x' and 'y' coordinates and
    only have dimensions of them. This function uses 'lat' and 'lon'
    coordinates to generate a scripfile.

    Parameters
    ----------
    source_grid_file : str
        input smb grid file

    source_grid_scripfile : str
        output scrip file of the input smb data
    """

    ds = xr.open_dataset(source_grid_file)
    out_file = netCDF4.Dataset(source_grid_scripfile, 'w')

    if "rlon" in ds and "rlat" in ds:  # this is for RACMO's rotated-pole grid
        nx = ds.sizes["rlon"]
        ny = ds.sizes["rlat"]
    else:
        nx = ds.sizes["x"]
        ny = ds.sizes["y"]
    units = 'degrees'

    grid_size = nx * ny

    out_file.createDimension("grid_size", grid_size)
    out_file.createDimension("grid_corners", 4)
    out_file.createDimension("grid_rank", 2)

    # Variables
    grid_center_lat = out_file.createVariable('grid_center_lat', 'f8',
                                              ('grid_size',))
    grid_center_lat.units = units
    grid_center_lon = out_file.createVariable('grid_center_lon', 'f8',
                                              ('grid_size',))
    grid_center_lon.units = units
    grid_corner_lat = out_file.createVariable('grid_corner_lat', 'f8',
                                              ('grid_size', 'grid_corners'))
    grid_corner_lat.units = units
    grid_corner_lon = out_file.createVariable('grid_corner_lon', 'f8',
                                              ('grid_size', 'grid_corners'))
    grid_corner_lon.units = units
    grid_imask = out_file.createVariable('grid_imask', 'i4', ('grid_size',))
    grid_imask.units = 'unitless'
    out_file.createVariable('grid_dims', 'i4', ('grid_rank',))

    out_file.variables['grid_center_lat'][:] = ds.lat.values.flat
    out_file.variables['grid_center_lon'][:] = ds.lon.values.flat
    out_file.variables['grid_dims'][:] = [nx, ny]
    out_file.variables['grid_imask'][:] = 1

    if 'lat_bnds' in ds and 'lon_bnds' in ds:
        lat_corner = ds.lat_bnds
        if "time" in lat_corner.dims:
            lat_corner = lat_corner.isel(time=0)

        lon_corner = ds.lon_bnds
        if "time" in lon_corner.dims:
            lon_corner = lon_corner.isel(time=0)

        lat_corner = lat_corner.values
        lon_corner = lon_corner.values
    else:  # this part is used for RACMO as it does not have lat_bnds & lon_bnds
        lat_corner = _unwrap_corners(interp_extrap_corners_2d(ds.lat.values))
        lon_corner = _unwrap_corners(interp_extrap_corners_2d(ds.lon.values))

    grid_corner_lat = lat_corner.reshape((grid_size, 4))
    grid_corner_lon = lon_corner.reshape((grid_size, 4))

    out_file.variables['grid_corner_lat'][:] = grid_corner_lat
    out_file.variables['grid_corner_lon'][:] = grid_corner_lon

    out_file.close()


def _unwrap_corners(in_field):
    """Turn a 2D array of corners into an array of rectangular mesh elements"""
    out_field = np.zeros(((in_field.shape[0] - 1) *
                          (in_field.shape[1] - 1), 4))
    # corners are counterclockwise
    out_field[:, 0] = in_field[0:-1, 0:-1].flat
    out_field[:, 1] = in_field[0:-1, 1:].flat
    out_field[:, 2] = in_field[1:, 1:].flat
    out_field[:, 3] = in_field[1:, 0:-1].flat

    return out_field
