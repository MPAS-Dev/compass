import glob
import os
import shutil

import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call

from compass.landice.tests.ismip7_forcing.create_mapfile import (
    build_mapping_file,
)
from compass.step import Step


class ProcessShelfCollapse(Step):
    """
    A step for processing the ISMIP7 ice shelf collapse mask (Path C).
    Remaps the annual 0/1 collapse mask from the ISMIP7 polar
    stereographic grid to the MALI unstructured mesh and renames
    variables to MALI conventions. The mask flags floating grid cells
    that collapse once excess meltwater (after firn air content
    depletion) exceeds 72.5 mm/yr for 10 consecutive years.
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_forcing.fracture.Fracture
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name="process_shelf_collapse")

    def setup(self):
        """
        Set up this step of the test case
        """
        config = self.config
        section = config["ismip7"]
        base_path_mali = section.get("base_path_mali")
        mali_mesh_file = section.get("mali_mesh_file")

        self.add_input_file(filename=mali_mesh_file,
                            target=os.path.join(base_path_mali,
                                                mali_mesh_file))

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config

        section = config["ismip7"]
        base_path_ismip7 = section.get("base_path_ismip7")
        mali_mesh_name = section.get("mali_mesh_name")
        mali_mesh_file = section.get("mali_mesh_file")
        model = section.get("model")
        scenario = section.get("scenario")
        output_base_path = section.get("output_base_path")
        ice_sheet = section.get("ice_sheet")

        section = config["ismip7_fracture"]
        method_remap = section.get("method_remap_shelf_collapse")
        version = section.get("version")
        start_year = section.getint("start_year")
        end_year = section.getint("end_year")

        # Skip this pathway if no remapping method is requested
        if method_remap.lower() == "none":
            logger.info("method_remap_shelf_collapse is None; skipping ice "
                        "shelf collapse mask (Path C) processing.")
            return

        # Discover the ice shelf collapse mask file
        input_path = os.path.join(base_path_ismip7, "fracture", version)
        file_pattern = "ice_shelf_collapse_mask_*.nc"
        all_files = sorted(glob.glob(os.path.join(input_path, file_pattern)))

        if not all_files:
            raise FileNotFoundError(
                f"No ice shelf collapse mask file found matching pattern:\n"
                f"  {os.path.join(input_path, file_pattern)}")
        if len(all_files) > 1:
            raise ValueError(
                f"Expected a single ice shelf collapse mask file but found "
                f"{len(all_files)}:\n  " + "\n  ".join(all_files))

        input_file = all_files[0]
        basename = os.path.basename(input_file)
        logger.info(f"Processing ice shelf collapse mask: {basename}")

        # Build mapping file. neareststod preserves the 0/1 mask values.
        mapping_file = (f"map_ismip7_{ice_sheet}_fracture_to_"
                        f"{mali_mesh_name}_{method_remap}.nc")

        if not os.path.exists(mapping_file):
            logger.info("Building mapping file for the collapse mask grid...")
            build_mapping_file(config, logger,
                               input_file, mapping_file,
                               mali_mesh_file=mali_mesh_file,
                               method_remap=method_remap)

        # Remap the collapse mask onto the MALI mesh
        remapped_file = f"remapped_{basename}"
        if not os.path.exists(remapped_file):
            logger.info(f"Remapping: {basename}")
            args = ["ncremap",
                    "-i", input_file,
                    "-o", remapped_file,
                    "-m", mapping_file,
                    "-v", "mask"]
            check_call(args, logger=logger)

        # Combine time slice and rename to MALI conventions
        logger.info("Renaming variables to MALI conventions...")
        output_file = f"{mali_mesh_name}_{basename}"
        self._rename_to_mali_vars(remapped_file, output_file,
                                  start_year, end_year)

        # Clean up temporary remapped file
        if os.path.exists(remapped_file):
            os.remove(remapped_file)

        # Place output in the appropriate directory
        output_path = os.path.join(output_base_path, "shelf_collapse",
                                   f"{model}_{scenario}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        dst = os.path.join(output_path, output_file)
        shutil.copy(output_file, dst)

        logger.info(f"Done. Output: {dst}")

    def _rename_to_mali_vars(self, remapped_file, output_file,
                             start_year, end_year):
        """
        Rename dimensions/variables of the remapped collapse mask to MALI
        conventions, restrict to the requested year range, round the mask
        to 0/1, and add an ``xtime`` variable.

        Parameters
        ----------
        remapped_file : str
            Collapse mask remapped onto the MALI mesh

        output_file : str
            Output file with MALI variable/dimension names

        start_year : int
            First year (inclusive) to retain

        end_year : int
            Last year (inclusive) to retain
        """
        # The collapse mask time coordinate has units="year" (integer years),
        # which is not CF-compliant, so disable time decoding.
        ds = xr.open_dataset(remapped_file, decode_times=False)

        # Rename dimensions to MALI conventions
        rename_dims = {}
        if "ncol" in ds.dims:
            rename_dims["ncol"] = "nCells"
        if "time" in ds.dims:
            rename_dims["time"] = "Time"
        if rename_dims:
            ds = ds.rename(rename_dims)

        # Rename variable
        if "mask" in ds:
            ds = ds.rename({"mask": "calvingMask"})

        # Restrict to the requested year range
        years = ds["time"].values.astype(int)
        keep = (years >= start_year) & (years <= end_year)
        ds = ds.isel(Time=keep)
        years = years[keep]

        # Round the remapped mask to 0/1 and store as integers. Ice shelves
        # collapse on January 1st, so the mask is applied at the start of
        # each year.
        calving_mask = (ds["calvingMask"] >= 0.5).astype(int)
        ds["calvingMask"] = calving_mask

        # Add xtime variable, one entry per year at January 1st
        xtime = [f"{int(yr):04d}-01-01_00:00:00".ljust(64) for yr in years]
        ds["xtime"] = ("Time", xtime)
        ds["xtime"] = ds.xtime.astype("S")

        ds["calvingMask"].attrs = {
            "long_name": "ice shelf collapse mask (1 = collapse)",
        }

        # Drop auxiliary variables carried over from remapping
        vars_to_drop = [v for v in ["lat_vertices", "lon_vertices", "lat",
                                    "lon", "area", "time"]
                        if v in ds]
        if vars_to_drop:
            ds = ds.drop_vars(vars_to_drop)

        write_netcdf(ds, output_file)
        ds.close()
