import os
import shutil
import subprocess
import netCDF4
import xarray as xr
from mpas_tools.scrip.from_mpas import scrip_from_mpas
from mpas_tools.logging import check_call


# function that creates a mapping file from ismip6 grid to mali mesh
def build_mapping_file(config, cores, logger, ismip6_grid_file,
                       mapping_file, mali_mesh_file=None,
                       method_remap=None):
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
        The MALI mesh file if mapping file does not exist
    method_remap : str, optional
        Remapping method used in building a mapping file
    """

    if mali_mesh_file is None:
        raise ValueError("Mapping file does not exist. To build one, Mali "
                         "mesh file with '-f' should be provided. "
                         "Type --help for info")

    ismip6_scripfile = "temp_ismip6_8km_scrip.nc"
    mali_scripfile = "temp_mali_scrip.nc"
    ismip6_projection = "ais-bedmap2"

    # create the ismip6 scripfile if mapping file does not exist
    # this is the projection of ismip6 data for Antarctica
    print("Mapping file does not exist. Building one based on the "
          "input/ouptut meshes")
    print("Creating temporary scripfiles for ismip6 grid and mali mesh...")

    # create a scripfile for the atmosphere forcing data
    create_atm_scrip(ismip6_grid_file, ismip6_scripfile)

    # create a MALI mesh scripfile if mapping file does not exist
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
    args = parallel_executable.split(' ')
    args.extend(["-n", f"{cores}",
                 "ESMF_RegridWeightGen",
                 "-s", ismip6_scripfile,
                 "-d", mali_scripfile,
                 "-w", mapping_file,
                 "-m", method_remap,
                 "-i", "-64bit_offset",
                 "--dst_regional", "--src_regional"])

    check_call(args, logger)

    # remove the temporary scripfiles once the mapping file is generated
    print("Removing the temporary mesh and scripfiles...")
    os.remove(ismip6_scripfile)
    os.remove(mali_scripfile)
    os.remove(mali_mesh_copy)


def create_atm_scrip(ismip6_grid_file, ismip6_scripfile):
    """
    create a scripfile for the ismip6 atmospheric forcing data.
    Note: the atmospheric forcing data do not have 'x' and 'y' coordinates and
    only have dimensions of them. This function uses 'lat' and 'lon'
    coordinates to generate a scripfile.

    Parameters
    ----------
    ismip6_grid_file : str
        input ismip6 grid file

    ismip6_scripfile : str
        outputput ismip6 scrip file
    """

    ds = xr.open_dataset(ismip6_grid_file)
    out_file = netCDF4.Dataset(ismip6_scripfile, 'w')

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

    lat_corner = ds.lat_bnds
    if "time" in lat_corner.dims:
        lat_corner = lat_corner.isel(time=0)

    grid_corner_lat = lat_corner.values.reshape((grid_size, 4))

    lon_corner = ds.lon_bnds
    if "time" in lon_corner.dims:
        lon_corner = lon_corner.isel(time=0)

    grid_corner_lon = lon_corner.values.reshape((grid_size, 4))

    out_file.variables['grid_corner_lat'][:] = grid_corner_lat
    out_file.variables['grid_corner_lon'][:] = grid_corner_lon

    out_file.close()
