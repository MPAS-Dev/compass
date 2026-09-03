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
from compass.landice.tests.ismip7_forcing.ice_sheet_params import get_params
from compass.step import Step


class ProcessThermalForcing(Step):
    """
    A step for processing ISMIP7 ocean thermal forcing (tf) data.
    For AIS: Remaps annual 3D thermal forcing from the ISMIP7 8km polar
    stereographic grid to the MALI unstructured mesh, preserving
    the 30 vertical ocean layers.
    For GrIS: Remaps monthly 2D thermal forcing from the ISMIP7 1km
    grid to the MALI unstructured mesh.
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_forcing.ocean_thermal.OceanThermal  # noqa
            The test case this step belongs to
        """
        super().__init__(test_case=test_case,
                         name="process_thermal_forcing")

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
        config = self.config
        section = config["ismip7"]

        # Check if we should process climatology data
        if section.getboolean("process_ocean_climatology"):
            self._run_climatology()

        # Check if we should process scenario (time-varying) data
        if section.getboolean("process_ocean_thermal"):
            self._run_scenario()

    def _run_scenario(self):
        """
        Process time-varying ocean thermal forcing from an ESM
        (e.g., CESM2-WACCM historical or ssp585).
        """
        logger = self.logger
        config = self.config
        params = get_params(config)

        section = config["ismip7"]
        base_path_ismip7 = section.get("base_path_ismip7")
        mali_mesh_name = section.get("mali_mesh_name")
        mali_mesh_file = section.get("mali_mesh_file")
        model = section.get("model")
        scenario = section.get("scenario")
        output_base_path = section.get("output_base_path")
        ice_sheet = section.get("ice_sheet")

        section = config["ismip7_ocean_thermal"]
        method_remap = section.get("method_remap")
        start_year = section.getint("start_year")
        end_year = section.getint("end_year")

        # Discover input files
        prefix = params['prefix']
        ocean_version = params['ocean_version']
        ocean_grid = params['ocean_grid']
        ocean_3d = params['ocean_3d']
        if params['ocean_model'] is not None:
            forcing_group = scenario
            model = params['ocean_model']
        else:
            forcing_group = f"{model}_{scenario}"
        input_path = os.path.join(base_path_ismip7, "ocean", "tf",
                                  ocean_version)
        file_pattern = (f"tf_{prefix}_{model}_{scenario}_"
                        f"{ocean_grid}_{ocean_version}_*.nc")
        all_files = sorted(glob.glob(os.path.join(input_path, file_pattern)))

        if not all_files:
            raise FileNotFoundError(
                f"No ocean thermal forcing files found matching pattern:\n"
                f"  {os.path.join(input_path, file_pattern)}")

        # Filter to files that overlap with the requested year range.
        # AIS files are named with decade ranges (e.g., 1850-1859).
        # GrIS files are named with single years (e.g., 2015).
        input_files = []
        for f in all_files:
            # Extract year range from filename (last part before .nc)
            year_str = os.path.basename(f).split("_")[-1].replace(".nc", "")
            parts = year_str.split("-")
            file_start = int(parts[0])
            file_end = int(parts[-1])  # same as start for single-year files
            if file_end >= start_year and file_start <= end_year:
                input_files.append(f)

        if not input_files:
            raise FileNotFoundError(
                f"No ocean thermal forcing files for year range "
                f"{start_year}-{end_year}")

        logger.info(f"Found {len(input_files)} ocean thermal forcing files "
                    f"overlapping years {start_year}-{end_year}")

        # Build mapping file using the first input file as grid template.
        mapping_file = (f"map_ismip7_{ice_sheet}_ocean_to_"
                        f"{mali_mesh_name}_{method_remap}.nc")

        if not os.path.exists(mapping_file):
            logger.info("Building mapping file for ocean grid...")
            build_mapping_file(config, logger,
                               input_files[0], mapping_file,
                               mali_mesh_file=mali_mesh_file,
                               method_remap=method_remap)

        # Remap each decade file
        remapped_files = []
        for input_file in input_files:
            basename = os.path.basename(input_file)
            remapped_file = f"remapped_{basename}"
            remapped_files.append(remapped_file)

            if os.path.exists(remapped_file):
                logger.info(f"  Remapped file exists, skipping: {basename}")
                continue

            # Extrapolate fill values on source grid before remapping
            # so they don't pollute neighboring cells during interpolation
            extrap_file = f"extrap_{basename}"
            if not os.path.exists(extrap_file):
                self._extrapolate_source(input_file, extrap_file, "tf",
                                         logger)

            logger.info(f"  Remapping: {basename}")
            args = ["ncremap",
                    "-i", extrap_file,
                    "-o", remapped_file,
                    "-m", mapping_file,
                    "-v", "tf"]

            check_call(args, logger=logger)

            # Clean up extrapolated source file
            os.remove(extrap_file)

        # Combine remapped files and rename to MALI conventions
        logger.info("Combining remapped files and renaming variables...")
        output_file = (f"{mali_mesh_name}_thermal_forcing_{model}_{scenario}_"
                       f"{start_year}-{end_year}.nc")

        if ocean_3d:
            self._combine_and_rename_3d(remapped_files, output_file,
                                        start_year, end_year)
        else:
            self._combine_and_rename_2d(remapped_files, output_file,
                                        start_year, end_year)

        # Clean up remapped files
        logger.info("Cleaning up temporary remapped files...")
        for f in remapped_files:
            if os.path.exists(f):
                os.remove(f)

        # Place output in appropriate directory
        output_path = os.path.join(output_base_path, forcing_group,
                                   "ocean_thermal_forcing")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        dst = os.path.join(output_path, output_file)
        shutil.copy(output_file, dst)

        logger.info(f"Done. Output: {dst}")

    def _run_climatology(self):
        """
        Process observational ocean thermal forcing climatology
        (e.g., Zhou et al. for AIS). This is a static 3D field with
        no time dimension.
        """
        logger = self.logger
        config = self.config

        section = config["ismip7"]
        mali_mesh_name = section.get("mali_mesh_name")
        mali_mesh_file = section.get("mali_mesh_file")
        output_base_path = section.get("output_base_path")
        ice_sheet = section.get("ice_sheet")

        section = config["ismip7_ocean_climatology"]
        method_remap = section.get("method_remap")
        base_path_climatology = section.get("base_path_climatology")
        version = 'v3'

        # Discover climatology TF file
        input_path = os.path.join(base_path_climatology, "tf", version)
        all_files = sorted(glob.glob(os.path.join(input_path, "tf_*.nc")))

        if not all_files:
            raise FileNotFoundError(
                f"No ocean climatology TF files found in:\n"
                f"  {input_path}")

        # Use the first (and likely only) file
        input_file = all_files[0]
        logger.info(f"Processing ocean TF climatology: "
                    f"{os.path.basename(input_file)}")

        # Build mapping file using the climatology file as grid template.
        mapping_file = (f"map_ismip7_{ice_sheet}_ocean_to_"
                        f"{mali_mesh_name}_{method_remap}.nc")

        if not os.path.exists(mapping_file):
            logger.info("Building mapping file for ocean grid...")
            build_mapping_file(config, logger,
                               input_file, mapping_file,
                               mali_mesh_file=mali_mesh_file,
                               method_remap=method_remap)

        # Extrapolate and remap
        basename = os.path.basename(input_file)
        remapped_file = f"remapped_{basename}"

        if not os.path.exists(remapped_file):
            extrap_file = f"extrap_{basename}"
            if not os.path.exists(extrap_file):
                self._extrapolate_source(input_file, extrap_file, "tf",
                                         logger)

            logger.info(f"  Remapping: {basename}")
            args = ["ncremap",
                    "-i", extrap_file,
                    "-o", remapped_file,
                    "-m", mapping_file,
                    "-v", "tf"]

            check_call(args, logger=logger)

            # Clean up extrapolated source file
            os.remove(extrap_file)

        # Rename to MALI conventions
        logger.info("Renaming variables to MALI conventions...")
        output_file = (f"{mali_mesh_name}_thermal_forcing_climatology_"
                       f"{version}.nc")

        self._rename_climatology_3d(remapped_file, output_file)

        # Clean up remapped file
        if os.path.exists(remapped_file):
            os.remove(remapped_file)

        # Place output in appropriate directory
        output_path = os.path.join(output_base_path, "ocean_thermal_forcing",
                                   "climatology")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        dst = os.path.join(output_path, output_file)
        shutil.copy(output_file, dst)

        logger.info(f"Done. Output: {dst}")

    def _combine_and_rename_3d(self, remapped_files, output_file,
                               start_year, end_year):
        """
        Combine decade-spanning remapped files (AIS), subset to the
        requested year range, and rename variables/dimensions to MALI
        conventions for 3D thermal forcing.

        Parameters
        ----------
        remapped_files : list of str
            List of remapped NetCDF file paths

        output_file : str
            Output file path

        start_year : int
            First year to include in output

        end_year : int
            Last year to include in output
        """
        ds = xr.open_mfdataset(remapped_files, concat_dim="time",
                               combine="nested", engine="netcdf4",
                               drop_variables="time_bnds")

        # Subset to requested year range
        years = ds.time.dt.year
        ds = ds.sel(time=(years >= start_year) & (years <= end_year))

        # Extract z coordinate and bounds before renaming
        z_ocean = ds["z"]
        z_bnds = ds["z_bnds"]
        if "time" in z_bnds.dims:
            z_bnds = z_bnds.isel(time=0)

        # Rename dimensions to MALI conventions
        rename_dims = {}
        if "time" in ds.dims:
            rename_dims["time"] = "Time"
        if "ncol" in ds.dims:
            rename_dims["ncol"] = "nCells"
        if "z" in ds.dims:
            rename_dims["z"] = "nISMIP6OceanLayers"
        if "bnds" in ds.dims:
            rename_dims["bnds"] = "TWO"
        ds = ds.rename(rename_dims)

        # Rename thermal forcing variable
        if "tf" in ds:
            ds = ds.rename({"tf": "ismip6shelfMelt_3dThermalForcing"})

        # Set z coordinate and bounds as MALI-named variables
        ds["ismip6shelfMelt_zOcean"] = (
            "nISMIP6OceanLayers", z_ocean.values)
        ds["ismip6shelfMelt_zBndsOcean"] = (
            ("TWO", "nISMIP6OceanLayers"), z_bnds.values.T)

        # Transpose thermal forcing to MALI dimension order
        # Registry: nISMIP6OceanLayers nCells Time (Fortran order)
        # NetCDF (C order): Time, nCells, nISMIP6OceanLayers
        ds["ismip6shelfMelt_3dThermalForcing"] = \
            ds["ismip6shelfMelt_3dThermalForcing"].transpose(
                "Time", "nCells", "nISMIP6OceanLayers")

        # Ensure double precision for MALI compatibility
        ds["ismip6shelfMelt_3dThermalForcing"] = \
            ds["ismip6shelfMelt_3dThermalForcing"].astype(float)

        # Add xtime variable with annual timestamps
        xtime = []
        for t_index in range(ds.sizes["Time"]):
            date = ds.Time[t_index]
            yr = int(date.dt.year.values)
            date_str = f"{yr:04d}-01-01_00:00:00".ljust(64)
            xtime.append(date_str)

        ds["xtime"] = ("Time", xtime)
        ds["xtime"] = ds.xtime.astype("S")

        # Set attributes
        ds["ismip6shelfMelt_3dThermalForcing"].attrs = {
            "long_name": "thermal forcing for ISMIP6 ice-shelf "
                         "melting method",
            "units": "degC",
        }
        # Remove stale encoding (e.g. 'coordinates' from ncremap)
        ds["ismip6shelfMelt_3dThermalForcing"].encoding.clear()
        ds["ismip6shelfMelt_zOcean"].attrs = {
            "long_name": "depth coordinate for ocean thermal forcing",
            "units": "m",
        }
        ds["ismip6shelfMelt_zBndsOcean"].attrs = {
            "long_name": "bounds for ISMIP6 ocean layers",
            "units": "m",
        }

        # Drop auxiliary variables from remapping
        vars_to_drop = [v for v in ["lon", "lon_vertices", "lat",
                                    "lat_vertices", "lon_bnds", "lat_bnds",
                                    "area", "z_bnds", "time_bnds",
                                    "x_bnds", "y_bnds"]
                        if v in ds]
        if vars_to_drop:
            ds = ds.drop_vars(vars_to_drop)

        # Also drop the renamed z coordinate if it persists
        if "nISMIP6OceanLayers" in ds.coords:
            ds = ds.drop_vars("nISMIP6OceanLayers")

        # Drop Time coordinate values (keep as dimension only)
        if "Time" in ds.coords:
            ds = ds.drop_vars("Time")

        write_netcdf(ds, output_file)

    def _combine_and_rename_2d(self, remapped_files, output_file,
                               start_year, end_year):
        """
        Combine yearly remapped files (GrIS), subset to the requested
        year range, and rename variables/dimensions to MALI conventions
        for 2D thermal forcing.

        Parameters
        ----------
        remapped_files : list of str
            List of remapped NetCDF file paths

        output_file : str
            Output file path

        start_year : int
            First year to include in output

        end_year : int
            Last year to include in output
        """
        ds = xr.open_mfdataset(remapped_files, concat_dim="time",
                               combine="nested", engine="netcdf4",
                               drop_variables="time_bnds")

        # Subset to requested year range
        years = ds.time.dt.year
        ds = ds.sel(time=(years >= start_year) & (years <= end_year))

        # Rename dimensions to MALI conventions
        rename_dims = {}
        if "time" in ds.dims:
            rename_dims["time"] = "Time"
        if "ncol" in ds.dims:
            rename_dims["ncol"] = "nCells"
        if rename_dims:
            ds = ds.rename(rename_dims)

        # Rename thermal forcing variable
        if "tf" in ds:
            ds = ds.rename({"tf": "ismip6_2dThermalForcing"})

        # Ensure double precision for MALI compatibility
        ds["ismip6_2dThermalForcing"] = \
            ds["ismip6_2dThermalForcing"].astype(float)

        # Add xtime variable with monthly timestamps
        # ISMIP7 files encode time at mid-month (e.g., Jan 15) but
        # this represents forcing for the full month (Jan 1-31).
        # MALI needs xtime at the start of each forcing interval.
        xtime = []
        for t_index in range(ds.sizes["Time"]):
            date = ds.Time[t_index]
            yr = int(date.dt.year.values)
            mo = int(date.dt.month.values)
            date_str = f"{yr:04d}-{mo:02d}-01_00:00:00".ljust(64)
            xtime.append(date_str)

        ds["xtime"] = ("Time", xtime)
        ds["xtime"] = ds.xtime.astype("S")

        # Set attributes
        ds["ismip6_2dThermalForcing"].attrs = {
            "long_name": "2D thermal forcing for ISMIP6 ice-shelf "
                         "melting parameterization",
            "units": "degC",
        }
        # Remove stale encoding (e.g. 'coordinates' from ncremap)
        ds["ismip6_2dThermalForcing"].encoding.clear()

        # Drop auxiliary variables from remapping
        vars_to_drop = [v for v in ["lon", "lon_vertices", "lat",
                                    "lat_vertices", "area",
                                    "time_bnds", "x_bnds", "y_bnds"]
                        if v in ds]
        if vars_to_drop:
            ds = ds.drop_vars(vars_to_drop)

        # Drop Time coordinate values (keep as dimension only)
        if "Time" in ds.coords:
            ds = ds.drop_vars("Time")

        write_netcdf(ds, output_file)

    def _rename_climatology_3d(self, remapped_file, output_file):
        """
        Rename dimensions and variables in a remapped 3D climatology
        file (no time dimension) to MALI conventions.

        Parameters
        ----------
        remapped_file : str
            Path to the remapped NetCDF file

        output_file : str
            Output file path
        """
        ds = xr.open_dataset(remapped_file, engine="netcdf4")

        # Extract z coordinate and bounds before renaming
        z_ocean = ds["z"]
        z_bnds = ds["z_bnds"]

        # Rename dimensions to MALI conventions
        rename_dims = {}
        if "ncol" in ds.dims:
            rename_dims["ncol"] = "nCells"
        if "z" in ds.dims:
            rename_dims["z"] = "nISMIP6OceanLayers"
        if "bnds" in ds.dims:
            rename_dims["bnds"] = "TWO"
        if rename_dims:
            ds = ds.rename(rename_dims)

        # Rename thermal forcing variable
        if "tf" in ds:
            ds = ds.rename({"tf": "ismip6shelfMelt_3dThermalForcing"})

        ds["ismip6shelfMelt_3dThermalForcing"] = \
            ds["ismip6shelfMelt_3dThermalForcing"].expand_dims("Time", axis=0)

        # Set z coordinate and bounds as MALI-named variables
        ds["ismip6shelfMelt_zOcean"] = (
            "nISMIP6OceanLayers", z_ocean.values)
        ds["ismip6shelfMelt_zBndsOcean"] = (
            ("TWO", "nISMIP6OceanLayers"), z_bnds.values.T)

        # Transpose thermal forcing to MALI dimension order
        # NetCDF (C order): nCells, nISMIP6OceanLayers
        ds["ismip6shelfMelt_3dThermalForcing"] = \
            ds["ismip6shelfMelt_3dThermalForcing"].transpose(
                "Time", "nCells", "nISMIP6OceanLayers")

        # Ensure double precision for MALI compatibility
        ds["ismip6shelfMelt_3dThermalForcing"] = \
            ds["ismip6shelfMelt_3dThermalForcing"].astype(float)

        # Set attributes
        ds["ismip6shelfMelt_3dThermalForcing"].attrs = {
            "long_name": "thermal forcing for ISMIP6 ice-shelf "
                         "melting method",
            "units": "degC",
        }
        ds["ismip6shelfMelt_3dThermalForcing"].encoding.clear()
        ds["ismip6shelfMelt_zOcean"].attrs = {
            "long_name": "depth coordinate for ocean thermal forcing",
            "units": "m",
        }
        ds["ismip6shelfMelt_zBndsOcean"].attrs = {
            "long_name": "bounds for ISMIP6 ocean layers",
            "units": "m",
        }

        # Drop auxiliary variables from remapping
        vars_to_drop = [v for v in ["lon", "lon_vertices", "lat",
                                    "lat_vertices", "lon_bnds", "lat_bnds",
                                    "area", "z_bnds", "time_bnds",
                                    "x_bnds", "y_bnds"]
                        if v in ds]
        if vars_to_drop:
            ds = ds.drop_vars(vars_to_drop)

        # Drop the z coordinate if it persists
        if "nISMIP6OceanLayers" in ds.coords:
            ds = ds.drop_vars("nISMIP6OceanLayers")

        write_netcdf(ds, output_file)

    def _extrapolate_source(self, input_file, output_file, varname, logger):
        """
        Extrapolate fill/missing values on the source polar stereographic
        grid using nearest-neighbor interpolation from valid cells. This
        must be done before remapping so that fill values don't contaminate
        the interpolation stencil.

        Parameters
        ----------
        input_file : str
            Path to the input NetCDF file on the source grid

        output_file : str
            Path to write the extrapolated file

        varname : str
            Name of the variable to extrapolate (e.g., "tf")

        logger : logging.Logger
            Logger for status messages
        """
        logger.info(f"    Extrapolating fill values on source grid: "
                    f"{os.path.basename(input_file)}")

        ds = xr.open_dataset(input_file, engine="netcdf4")
        data = ds[varname]

        # Process each time step (and z level if 3D)
        # Source files have dims like (time, z, y, x) or (time, y, x)
        values = data.values.copy()
        non_spatial_shape = values.shape[:-2]  # (time,) or (time, z)

        # Use distance_transform_edt with return_indices to find the
        # nearest valid cell index for each invalid cell. This is O(n)
        # on the grid and much faster than KD-tree approaches.
        for idx in np.ndindex(non_spatial_shape):
            slab = values[idx]  # shape (ny, nx)
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

        # Remove _FillValue encoding so output has no masked values
        if "_FillValue" in ds[varname].encoding:
            del ds[varname].encoding["_FillValue"]

        write_netcdf(ds, output_file)
