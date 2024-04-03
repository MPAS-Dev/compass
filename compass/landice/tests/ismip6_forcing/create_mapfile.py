import os
import shutil

import netCDF4
import xarray as xr
from mpas_tools.logging import check_call
from mpas_tools.scrip.from_mpas import scrip_from_mpas
from pyremap.descriptor.utility import (
    create_scrip,
    interp_extrap_corners_2d,
    unwrap_corners,
)


def build_mapping_file(config, cores, logger, ismip6_grid_file, mapping_file,
                       scrip_from_latlon=True, mali_mesh_file=None,
                       method_remap=None):
    """
    Build a mapping file if it does not exist.
    Mapping file is then used to remap the ismip6 source file in polarstero
    coordinate to unstructured mali mesh

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

    scrip_from_latlon : bool, optional
        whether to use the `lat`/`lon` coordinates to create the SCRIP file
        for the `ismip6_grid_file` passed into the function

    mali_mesh_file : str, optional
        The MALI mesh file is used if mapping file does not exist

    method_remap : str, optional
        Remapping method used in building a mapping file
    """

    if os.path.exists(mapping_file):
        logger.info("Mapping file exists. Not building a new one.")
        return

    # create the scrip files if mapping file does not exist
    logger.info("Mapping file does not exist. Building one based on the"
                " input/output meshes")
    logger.info("Creating temporary scrip files for source and "
                "destination grids...")

    if mali_mesh_file is None:
        raise ValueError("Mapping file does not exist. To build one, Mali "
                         "mesh file with '-f' should be provided. "
                         "Type --help for info")

    # name temporary scrip files that will be used to build mapping file
    source_grid_scripfile = "temp_source_scrip.nc"
    mali_scripfile = "temp_mali_scrip.nc"
    # this is the projection of ismip6 data for Antarctica
    ismip6_projection = "ais-bedmap2"

    # create the scrip file for the forcing dataset
    if scrip_from_latlon:
        create_scrip_from_latlon(ismip6_grid_file, source_grid_scripfile)
    else:
        args = ["create_SCRIP_file_from_planar_rectangular_grid.py",
                "--input", ismip6_grid_file,
                "--scrip", source_grid_scripfile,
                "--proj", ismip6_projection,
                "--rank", "2"]

        check_call(args, logger=logger)

    # create a MALI mesh scripfile
    # make sure the mali mesh file uses the longitude convention of [0 2pi]
    # make changes on a duplicated file to avoid making changes to the
    # original mesh file

    mali_mesh_copy = f"{mali_mesh_file}_copy"
    shutil.copy(mali_mesh_file, f"{mali_mesh_file}_copy")

    args = ["set_lat_lon_fields_in_planar_grid.py",
            "--file", mali_mesh_copy,
            "--proj", ismip6_projection]

    check_call(args, logger=logger)

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
                 "-s", source_grid_scripfile,
                 "-d", mali_scripfile,
                 "-w", mapping_file,
                 "-m", method_remap,
                 "-i", "-64bit_offset",
                 "--dst_regional", "--src_regional"])

    check_call(args, logger=logger)

    # remove the temporary scripfiles once the mapping file is generated
    logger.info("Removing the temporary mesh and scripfiles...")
    os.remove(source_grid_scripfile)
    os.remove(mali_scripfile)
    os.remove(mali_mesh_copy)


def create_scrip_from_latlon(source_grid_file, source_grid_scripfile):
    """
    Create a scripfile based on the `lat`/`lon` coordinates of a source
    dataset.

    This function is needed, c.f. the scrip utility in the MPAS-Tools repo
    (i.e. `create_SCRIP_file_from_planar_rectangular_grid.py`), when a dataset
    does not have `x`/`y` coordinates to generate the scrip file from. This is
    the case for the atmospheric forcing datasets from ISMIP6
    and for RACMO products.

    Parameters
    ----------
    source_grid_file : str
        input dataset (with `lat`/`lon` coords) to generate a scrip file for

    source_grid_scripfile : str
        output scrip file of the input smb data
    """

    ds = xr.open_dataset(source_grid_file)
    out_file = netCDF4.Dataset(source_grid_scripfile, 'w')

    # RACMO datasets, which use a rotated-pole grid, do not contain `x`/`y`
    # dimensions, instead use `rlat`/`rlon` dimensions to find `nx`/`ny`
    if "rlon" in ds and "rlat" in ds:
        nx = ds.sizes["rlon"]
        ny = ds.sizes["rlat"]
    else:
        nx = ds.sizes["x"]
        ny = ds.sizes["y"]

    grid_size = nx * ny

    # generate common variables used in scrip files
    create_scrip(out_file, grid_size, grid_corners=4, grid_rank=2,
                 units="degrees", meshName=source_grid_file)

    # place the information from our source dataset into the scrip dataset
    out_file.variables["grid_center_lat"][:] = ds.lat.values.flat
    out_file.variables["grid_center_lon"][:] = ds.lon.values.flat
    out_file.variables["grid_dims"][:] = [nx, ny]
    out_file.variables["grid_imask"][:] = 1

    # determine the corners of gricells
    if "lat_bnds" in ds and "lon_bnds" in ds:
        lat_corner = ds.lat_bnds
        if "time" in lat_corner.dims:
            lat_corner = lat_corner.isel(time=0)

        lon_corner = ds.lon_bnds
        if "time" in lon_corner.dims:
            lon_corner = lon_corner.isel(time=0)

        lat_corner = lat_corner.values
        lon_corner = lon_corner.values
    else:
        # RACMO datasets do not have `lat_bnds`/`lon_bnds` variables. Instead
        # the bounds of the RACMO dataset are approximated assuming the cell
        # centers are located at the midpoint b/w cell edges
        lat_corner = unwrap_corners(interp_extrap_corners_2d(ds.lat.values))
        lon_corner = unwrap_corners(interp_extrap_corners_2d(ds.lon.values))

    grid_corner_lat = lat_corner.reshape((grid_size, 4))
    grid_corner_lon = lon_corner.reshape((grid_size, 4))

    out_file.variables["grid_corner_lat"][:] = grid_corner_lat
    out_file.variables["grid_corner_lon"][:] = grid_corner_lon

    out_file.close()
