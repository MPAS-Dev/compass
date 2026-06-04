import os
import shutil

from mpas_tools.logging import check_call
from mpas_tools.scrip.from_mpas import scrip_from_mpas


def build_mapping_file(config, logger, ismip7_grid_file,
                       mapping_file, mali_mesh_file=None,
                       method_remap=None):
    """
    Build a mapping file for regridding from the ISMIP7 2km polar
    stereographic grid to the MALI unstructured mesh.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for the test case

    logger : logging.Logger
        A logger for output from the step

    ismip7_grid_file : str
        An ISMIP7 grid file (with x/y coordinates)

    mapping_file : str
        Output mapping file path

    mali_mesh_file : str, optional
        The MALI mesh file

    method_remap : str, optional
        Remapping method: 'bilinear', 'neareststod', or 'conserve'
    """

    if os.path.exists(mapping_file):
        logger.info("Mapping file exists. Not building a new one.")
        return

    logger.info("Mapping file does not exist. Building one based on the"
                " input/output meshes")

    if mali_mesh_file is None:
        raise ValueError("Mapping file does not exist. A MALI mesh file "
                         "must be provided to build one.")

    if method_remap is None:
        raise ValueError("Remapping method must be provided. "
                         "Options: 'bilinear', 'neareststod', 'conserve'.")

    # AIS polar stereographic projection (EPSG:3031)
    ismip7_projection = "ais-bedmap2"

    # name temporary scrip files
    source_grid_scripfile = "temp_source_scrip.nc"
    mali_scripfile = "temp_mali_scrip.nc"

    # create the scrip file for the ISMIP7 planar rectangular grid
    logger.info("Creating SCRIP file for ISMIP7 source grid...")
    args = ["create_scrip_file_from_planar_rectangular_grid",
            "--input", ismip7_grid_file,
            "--scrip", source_grid_scripfile,
            "--proj", ismip7_projection,
            "--rank", "2"]

    check_call(args, logger=logger)

    # create a MALI mesh scrip file
    logger.info("Creating SCRIP file for MALI mesh...")
    mali_mesh_copy = f"{mali_mesh_file}_copy"
    shutil.copy(mali_mesh_file, mali_mesh_copy)

    args = ["set_lat_lon_fields_in_planar_grid",
            "--file", mali_mesh_copy,
            "--proj", ismip7_projection]

    check_call(args, logger=logger)

    scrip_from_mpas(mali_mesh_copy, mali_scripfile)

    # create a mapping file using ESMF_RegridWeightGen
    logger.info(f"Creating mapping file with method: {method_remap}")

    section = config["ismip7_ais"]
    cores = section.getint("esmf_ntasks")

    parallel_executable = config.get("parallel", "parallel_executable")
    args = parallel_executable.split(" ")
    args.extend(["-n", f"{cores}",
                 "ESMF_RegridWeightGen",
                 "-s", source_grid_scripfile,
                 "-d", mali_scripfile,
                 "-w", mapping_file,
                 "-m", method_remap,
                 "-i", "-64bit_offset",
                 "--dst_regional", "--src_regional"])

    check_call(args, logger=logger)

    # clean up temporary files
    logger.info("Removing temporary scrip files...")
    os.remove(source_grid_scripfile)
    os.remove(mali_scripfile)
    os.remove(mali_mesh_copy)
