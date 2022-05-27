import os
import subprocess
from mpas_tools.scrip.from_mpas import scrip_from_mpas


# function that creates a mapping file from ismip6 grid to mali mesh
def build_mapping_file(ismip6_grid_file, mapping_file, mali_mesh_file=None,
                       method_remap=None):
    """
    Build a mapping file if it does not exist.

    Parameters
    ----------
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
        print("Mapping file exists. Not building a new one.")
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
    print("Mapping file does not exist. Building one based on the "
          "input/ouptut meshes")
    print("Creating temporary scripfiles for ismip6 grid and mali mesh...")

    args = ["create_SCRIP_file_from_planar_rectangular_grid.py",
            "--input", ismip6_grid_file,
            "--scrip", ismip6_scripfile,
            "--proj", ismip6_projection,
            "--rank", "2"]

    subprocess.check_call(args)

    # create a MALI mesh scripfile if mapping file does not exist
    scrip_from_mpas(mali_mesh_file, mali_scripfile)

    # create a mapping file using ESMF weight gen
    print(f"Creating a mapping file. Mapping method used: {method_remap}")

    if method_remap is None:
        raise ValueError("Desired remapping option should be provided with "
                         "--method. Available options are 'bilinear',"
                         "'neareststod', 'conserve'.")

    args = ["ESMF_RegridWeightGen",
            "-s", ismip6_scripfile,
            "-d", mali_scripfile,
            "-w", mapping_file,
            "-m", method_remap,
            "-i", "-64bit_offset",
            "--dst_regional", "--src_regional"]

    # include flag and input and output file names
    subprocess.check_call(args)

    # remove the temporary scripfiles once the mapping file is generated
    print("Removing the temporary scripfiles...")
    os.remove(ismip6_scripfile)
    os.remove(mali_scripfile)
