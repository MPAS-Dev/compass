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


class ProcessLakeProperties(Step):
    """
    A step for processing the ISMIP7 supraglacial lake properties (Path B).
    Remaps the annual mean lake depth and lake area fraction (from the
    Grau et al. (2025) parameterization) from the ISMIP7 polar
    stereographic grid to the MALI unstructured mesh.
    """

    # Source variable name -> MALI output variable name and attributes
    _variables = {
        "lake_depth": {
            "mali_name": "ismip7LakeDepth",
            "long_name": "mean supraglacial lake depth",
            "units": "m",
        },
        "fraction_lake_area": {
            "mali_name": "ismip7LakeAreaFraction",
            "long_name": "supraglacial lake area fraction",
            "units": "1",
        },
    }

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_forcing.fracture.Fracture
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name="process_lake_properties")

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
        method_remap = section.get("method_remap_lake_properties")
        version = section.get("version")
        start_year = section.getint("start_year")
        end_year = section.getint("end_year")

        # Skip this pathway if no remapping method is requested
        if method_remap.lower() == "none":
            logger.info("method_remap_lake_properties is None; skipping lake "
                        "properties (Path B) processing.")
            return

        # Discover the lake properties file
        input_path = os.path.join(base_path_ismip7, "fracture", version)
        file_pattern = "lake_properties_*.nc"
        all_files = sorted(glob.glob(os.path.join(input_path, file_pattern)))

        if not all_files:
            raise FileNotFoundError(
                f"No lake properties file found matching pattern:\n"
                f"  {os.path.join(input_path, file_pattern)}")
        if len(all_files) > 1:
            raise ValueError(
                f"Expected a single lake properties file but found "
                f"{len(all_files)}:\n  " + "\n  ".join(all_files))

        input_file = all_files[0]
        basename = os.path.basename(input_file)
        logger.info(f"Processing lake properties: {basename}")

        # Build mapping file. Lake properties are continuous fields, so
        # bilinear remapping is appropriate by default.
        mapping_file = (f"map_ismip7_{ice_sheet}_fracture_to_"
                        f"{mali_mesh_name}_{method_remap}.nc")

        if not os.path.exists(mapping_file):
            logger.info("Building mapping file for the lake properties "
                        "grid...")
            build_mapping_file(config, logger,
                               input_file, mapping_file,
                               mali_mesh_file=mali_mesh_file,
                               method_remap=method_remap)

        # Extrapolate fill values on the source grid before remapping so
        # they don't pollute neighboring cells during interpolation
        extrap_file = f"extrap_{basename}"
        self._extrapolate_source(input_file, extrap_file,
                                 list(self._variables.keys()), logger)

        # Remap both lake property variables onto the MALI mesh
        remapped_file = f"remapped_{basename}"
        logger.info(f"Remapping: {basename}")
        args = ["ncremap",
                "-i", extrap_file,
                "-o", remapped_file,
                "-m", mapping_file,
                "-v", ",".join(self._variables.keys())]
        check_call(args, logger=logger)

        # Rename to MALI conventions
        logger.info("Renaming variables to MALI conventions...")
        output_file = f"{mali_mesh_name}_{basename}"
        self._rename_to_mali_vars(remapped_file, output_file,
                                  start_year, end_year)

        # Clean up temporary files
        for f in [extrap_file, remapped_file]:
            if os.path.exists(f):
                os.remove(f)

        # Place output in the appropriate directory
        output_path = os.path.join(output_base_path, "lake_properties",
                                   f"{model}_{scenario}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        dst = os.path.join(output_path, output_file)
        shutil.copy(output_file, dst)

        logger.info(f"Done. Output: {dst}")

    def _rename_to_mali_vars(self, remapped_file, output_file,
                             start_year, end_year):
        """
        Rename dimensions/variables of the remapped lake properties to MALI
        conventions, restrict to the requested year range, and add an
        ``xtime`` variable.

        Parameters
        ----------
        remapped_file : str
            Lake properties remapped onto the MALI mesh

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

        rename_dims = {}
        if "ncol" in ds.dims:
            rename_dims["ncol"] = "nCells"
        if "time" in ds.dims:
            rename_dims["time"] = "Time"
        if rename_dims:
            ds = ds.rename(rename_dims)

        rename_vars = {src: info["mali_name"]
                       for src, info in self._variables.items()
                       if src in ds}
        ds = ds.rename(rename_vars)

        # Restrict to the requested year range
        years = ds["time"].values.astype(int)
        keep = (years >= start_year) & (years <= end_year)
        ds = ds.isel(Time=keep)
        years = years[keep]

        # Lake properties are annual fields applied at the start of each year
        xtime = [f"{int(yr):04d}-01-01_00:00:00".ljust(64) for yr in years]
        ds["xtime"] = ("Time", xtime)
        ds["xtime"] = ds.xtime.astype("S")

        for info in self._variables.values():
            mali_name = info["mali_name"]
            if mali_name in ds:
                ds[mali_name].attrs = {
                    "long_name": info["long_name"],
                    "units": info["units"],
                }

        vars_to_drop = [v for v in ["lat_vertices", "lon_vertices", "lat",
                                    "lon", "area", "time"]
                        if v in ds]
        if vars_to_drop:
            ds = ds.drop_vars(vars_to_drop)

        write_netcdf(ds, output_file)
        ds.close()

    def _extrapolate_source(self, input_file, output_file, varnames, logger):
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

        varnames : list of str
            Names of the variables to extrapolate

        logger : logging.Logger
            Logger for status messages
        """
        logger.info(f"    Extrapolating fill values on source grid: "
                    f"{os.path.basename(input_file)}")

        ds = xr.open_dataset(input_file, decode_times=False)

        for varname in varnames:
            data = ds[varname]
            values = data.values.copy()
            non_spatial_shape = values.shape[:-2]

            for idx in np.ndindex(non_spatial_shape):
                slab = values[idx]
                valid_mask = np.isfinite(slab)
                if valid_mask.all() or not valid_mask.any():
                    continue
                nearest_inds = distance_transform_edt(
                    ~valid_mask, return_distances=False,
                    return_indices=True)
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
