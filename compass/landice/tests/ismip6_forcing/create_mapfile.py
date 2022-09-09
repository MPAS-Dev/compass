import os
import shutil
import subprocess
from mpas_tools.scrip.from_mpas import scrip_from_mpas
from mpas_tools.logging import check_call


def build_mapping_file(config, cores, logger, ismip6_grid_file,
                       mapping_file, mali_mesh_file=None,
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
    mali_mesh_file : str, optional
        The MALI mesh file if mapping file does not exist
    method_remap : str, optional
        Remapping method used in building a mapping file
    """

    if os.path.exists(mapping_file):
        logger.info(f"Mapping file exists. Not building a new one.")
        return

    if mali_mesh_file is None:
        raise ValueError("Mapping file does not exist. To build one, Mali "
                         "mesh file with '-f' should be provided. "
                         "Type --help for info")

    ismip6_scripfile = "temp_ismip6_8km_scrip.nc"
    mali_scripfile = "temp_mali_scrip.nc"
    ismip6_projection = "ais-bedmap2"

    # create the ismip6 scripfile if mapping file does not exist
    # this is the projection of ismip6 data for Antarctica
    logger.info(f"Mapping file does not exist. Building one based on "
                f"the input/ouptut meshes")
    logger.info(f"Creating temporary scripfiles "
                f"for ismip6 grid and mali mesh...")

    args = ["create_SCRIP_file_from_planar_rectangular_grid.py",
            "--input", ismip6_grid_file,
            "--scrip", ismip6_scripfile,
            "--proj", ismip6_projection,
            "--rank", "2"]

    check_call(args, logger=logger)

    # create a MALI mesh scripfile if mapping file does not exist
    # make sure the mali mesh file uses the longitude convention of [0 2pi]
    # make changes on a duplicated file to avoid making changes to the
    # original mesh file

    mali_mesh_copy = f"{mali_mesh_file}_copy"
    shutil.copy(mali_mesh_file, f"{mali_mesh_file}_copy")

    args = ["set_lat_lon_fields_in_planar_grid.py",
            "--file", mali_mesh_copy,
            "--proj", ismip6_projection]

    check_call(args, logger=logger)

    scrip_from_mpas(mali_mesh_file, mali_scripfile)

    # create a mapping file using ESMF weight gen
    logger.info(f"Creating a mapping file... "
                f"Mapping method used: {method_remap}")

    if method_remap is None:
        raise ValueError("Desired remapping option should be provided with "
                         "--method. Available options are 'bilinear',"
                         "'neareststod', 'conserve'.")

    parallel_executable = config.get("parallel", "parallel_executable")
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
    logger.info(f"Removing the temporary mesh and scripfiles...")
    os.remove(ismip6_scripfile)
    os.remove(mali_scripfile)
    os.remove(mali_mesh_copy)
