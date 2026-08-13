import glob
import os
import shutil

import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from scipy.ndimage import distance_transform_edt

from compass.landice.tests.ismip7_forcing.create_mapfile import (
    build_mapping_file,
)
from compass.step import Step


class ProcessExcessMelt(Step):
    """
    A step for processing the ISMIP7 excess meltwater field (Path A).
    Remaps the annual excess melt (melt + rain after firn air content
    depletion) from the ISMIP7 polar stereographic grid to the MALI
    unstructured mesh.

    The excess melt file lacks ``x``/``y`` coordinate variables and its
    array is flipped along the y axis relative to the other fracture
    files (it was produced with CDO). This step reconstructs a source
    grid with ``x``/``y`` coordinates and the correct orientation before
    remapping.
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_forcing.fracture.Fracture
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name="process_excess_melt")

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
        method_remap = section.get("method_remap_excess_melt")
        version = section.get("version")
        start_year = section.getint("start_year")
        end_year = section.getint("end_year")

        # Skip this pathway if no remapping method is requested
        if method_remap.lower() == "none":
            logger.info("method_remap_excess_melt is None; skipping excess "
                        "melt (Path A) processing.")
            return

        # Discover the excess melt file
        input_path = os.path.join(base_path_ismip7, "fracture", version)
        file_pattern = "excess_melt_*.nc"
        all_files = sorted(glob.glob(os.path.join(input_path, file_pattern)))

        if not all_files:
            raise FileNotFoundError(
                f"No excess melt file found matching pattern:\n"
                f"  {os.path.join(input_path, file_pattern)}")
        if len(all_files) > 1:
            raise ValueError(
                f"Expected a single excess melt file but found "
                f"{len(all_files)}:\n  " + "\n  ".join(all_files))

        input_file = all_files[0]
        basename = os.path.basename(input_file)
        logger.info(f"Processing excess melt: {basename}")

        # Build a source file with x/y coordinates and correct orientation
        gridded_file = f"gridded_{basename}"
        self._prepare_source_grid(input_file, input_path, gridded_file,
                                  logger)

        # Build mapping file. Excess melt is a flux, so conservative
        # remapping is appropriate by default.
        mapping_file = (f"map_ismip7_{ice_sheet}_fracture_to_"
                        f"{mali_mesh_name}_{method_remap}.nc")

        if not os.path.exists(mapping_file):
            logger.info("Building mapping file for the excess melt grid...")
            build_mapping_file(config, logger,
                               gridded_file, mapping_file,
                               mali_mesh_file=mali_mesh_file,
                               method_remap=method_remap)

        # Extrapolate fill values on the source grid before remapping so
        # they don't pollute neighboring cells during interpolation
        extrap_file = f"extrap_{basename}"
        self._extrapolate_source(gridded_file, extrap_file, "excess_melt",
                                 logger)

        # Remap the excess melt onto the MALI mesh
        remapped_file = f"remapped_{basename}"
        logger.info(f"Remapping: {basename}")
        args = ["ncremap",
                "-i", extrap_file,
                "-o", remapped_file,
                "-m", mapping_file,
                "-v", "excess_melt"]
        check_call(args, logger=logger)

        # Rename to MALI conventions
        logger.info("Renaming variables to MALI conventions...")
        output_file = f"{mali_mesh_name}_{basename}"
        self._rename_to_mali_vars(remapped_file, output_file,
                                  start_year, end_year)

        # Clean up temporary files
        for f in [gridded_file, extrap_file, remapped_file]:
            if os.path.exists(f):
                os.remove(f)

        # Place output in the appropriate directory
        output_path = os.path.join(output_base_path, "excess_melt",
                                   f"{model}_{scenario}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        dst = os.path.join(output_path, output_file)
        shutil.copy(output_file, dst)

        logger.info(f"Done. Output: {dst}")

    def _prepare_source_grid(self, input_file, input_path, output_file,
                             logger):
        """
        Build a source file for the excess melt data with ``x``/``y``
        coordinate variables and the standard ISMIP7 orientation
        (south-to-north). The excess melt file only has 2D ``lat``/``lon``
        and its array is flipped along the y axis relative to the other
        fracture files, so ``x``/``y`` are borrowed from a sibling
        fracture file and the data are flipped to match.

        Parameters
        ----------
        input_file : str
            Path to the excess melt file

        input_path : str
            Directory containing the fracture forcing files

        output_file : str
            Path to write the reconstructed source file

        logger : logging.Logger
            Logger for status messages
        """
        # Find a sibling fracture file that has x/y coordinate variables
        grid_donor = None
        for pattern in ["lake_properties_*.nc",
                        "ice_shelf_collapse_mask_*.nc"]:
            for candidate in sorted(glob.glob(os.path.join(input_path,
                                                           pattern))):
                with xr.open_dataset(candidate) as dtest:
                    if "x" in dtest.variables and "y" in dtest.variables:
                        grid_donor = candidate
                        break
            if grid_donor is not None:
                break

        if grid_donor is None:
            raise FileNotFoundError(
                "Could not find a sibling fracture file with x/y "
                "coordinate variables to define the excess melt grid.")

        logger.info(f"Using grid from sibling file: "
                    f"{os.path.basename(grid_donor)}")

        with xr.open_dataset(grid_donor) as donor:
            x = donor["x"].values
            y = donor["y"].values
            donor_lat = donor["lat"].values

        ds = xr.open_dataset(input_file, decode_times=False)

        # Flip along the y axis to match the sibling grid orientation
        flipped_lat = ds["lat"].values[::-1, :]
        if np.nanmax(np.abs(flipped_lat - donor_lat)) > 1.0e-3:
            raise ValueError(
                "Excess melt grid does not match the sibling grid after a "
                "y-axis flip; orientation cannot be determined "
                "automatically.")

        excess = ds["excess_melt"].values[:, ::-1, :]
        years = ds["year"].values.astype(int)

        out = xr.Dataset()
        out["x"] = ("x", x)
        out["y"] = ("y", y)
        out["time"] = ("time", years)
        out["excess_melt"] = (("time", "y", "x"), excess)
        out["excess_melt"].attrs = dict(ds["excess_melt"].attrs)

        write_netcdf(out, output_file)
        ds.close()

    def _rename_to_mali_vars(self, remapped_file, output_file,
                             start_year, end_year):
        """
        Rename dimensions/variables of the remapped excess melt to MALI
        conventions, restrict to the requested year range, and add an
        ``xtime`` variable.

        Parameters
        ----------
        remapped_file : str
            Excess melt remapped onto the MALI mesh

        output_file : str
            Output file with MALI variable/dimension names

        start_year : int
            First year (inclusive) to retain

        end_year : int
            Last year (inclusive) to retain
        """
        # The time coordinate has units="year" (integer years), which is
        # not CF-compliant, so disable time decoding.
        ds = xr.open_dataset(remapped_file, decode_times=False)

        # Capture integer years before the time coordinate is renamed
        years = ds["time"].values.astype(int)

        rename_dims = {}
        if "ncol" in ds.dims:
            rename_dims["ncol"] = "nCells"
        if "time" in ds.dims:
            rename_dims["time"] = "Time"
        if rename_dims:
            ds = ds.rename(rename_dims)

        if "excess_melt" in ds:
            ds = ds.rename({"excess_melt": "ismip7ExcessMelt"})

        # Restrict to the requested year range
        keep = (years >= start_year) & (years <= end_year)
        ds = ds.isel(Time=keep)
        years = years[keep]

        # Excess melt is an annual field applied at the start of each year
        xtime = [f"{int(yr):04d}-01-01_00:00:00".ljust(64) for yr in years]
        ds["xtime"] = ("Time", xtime)
        ds["xtime"] = ds.xtime.astype("S")

        # Convert from mm w.e. yr-1 to SI units of kg m-2 s-1
        # (1 mm w.e. = 1 kg m-2; 365-day year, as used elsewhere in MALI)
        seconds_per_year = 365.0 * 24.0 * 3600.0
        ds["ismip7ExcessMelt"] = ds["ismip7ExcessMelt"] / seconds_per_year

        ds["ismip7ExcessMelt"].attrs = {
            "long_name": "excess meltwater after firn air content depletion",
            "units": "kg m-2 s-1",
        }

        vars_to_drop = [v for v in ["lat_vertices", "lon_vertices", "lat",
                                    "lon", "area", "Time"]
                        if v in ds]
        if vars_to_drop:
            ds = ds.drop_vars(vars_to_drop)

        write_netcdf(ds, output_file)
        ds.close()

    def _extrapolate_source(self, input_file, output_file, varname, logger):
        """
        Extrapolate fill/missing values on the source polar stereographic
        grid using nearest-neighbor via distance_transform_edt. This must
        be done before remapping so that fill values don't contaminate the
        interpolation stencil.

        Parameters
        ----------
        input_file : str
            Path to the input NetCDF file on the source grid

        output_file : str
            Path to write the extrapolated file

        varname : str
            Name of the variable to extrapolate

        logger : logging.Logger
            Logger for status messages
        """
        logger.info(f"    Extrapolating fill values on source grid: "
                    f"{os.path.basename(input_file)}")

        ds = xr.open_dataset(input_file, decode_times=False)
        data = ds[varname]

        values = data.values.copy()
        non_spatial_shape = values.shape[:-2]

        for idx in np.ndindex(non_spatial_shape):
            slab = values[idx]
            valid_mask = np.isfinite(slab)
            if valid_mask.all() or not valid_mask.any():
                continue
            nearest_inds = distance_transform_edt(
                ~valid_mask, return_distances=False, return_indices=True)
            invalid = ~valid_mask
            values[idx][invalid] = slab[
                nearest_inds[0, invalid],
                nearest_inds[1, invalid]]

        ds[varname] = (data.dims, values)
        ds[varname].attrs = data.attrs

        if "_FillValue" in ds[varname].encoding:
            del ds[varname].encoding["_FillValue"]

        write_netcdf(ds, output_file)
        ds.close()
