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


class ProcessThermalForcing(Step):
    """
    A step for processing ISMIP7 ocean thermal forcing (tf) data.
    Remaps annual 3D thermal forcing from the ISMIP7 8km polar
    stereographic grid to the MALI unstructured mesh, preserving
    the 30 vertical ocean layers.
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
        section = config["ismip7_ais"]
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

        section = config["ismip7_ais"]
        base_path_ismip7 = section.get("base_path_ismip7")
        mali_mesh_name = section.get("mali_mesh_name")
        mali_mesh_file = section.get("mali_mesh_file")
        model = section.get("model")
        scenario = section.get("scenario")
        output_base_path = section.get("output_base_path")

        section = config["ismip7_ais_ocean_thermal"]
        method_remap = section.get("method_remap")
        start_year = section.getint("start_year")
        end_year = section.getint("end_year")

        # Discover input files (decade-spanning files)
        input_path = os.path.join(base_path_ismip7, "ocean", "tf", "v3")
        file_pattern = (f"tf_AIS_{model}_{scenario}_"
                        f"ocean_v3_*.nc")
        all_files = sorted(glob.glob(os.path.join(input_path, file_pattern)))

        if not all_files:
            raise FileNotFoundError(
                f"No ocean thermal forcing files found matching pattern:\n"
                f"  {os.path.join(input_path, file_pattern)}")

        # Filter to files that overlap with the requested year range.
        # Files are named with decade ranges (e.g., 1850-1859).
        input_files = []
        for f in all_files:
            # Extract year range from filename (last part before .nc)
            year_str = os.path.basename(f).split("_")[-1].replace(".nc", "")
            parts = year_str.split("-")
            file_start = int(parts[0])
            file_end = int(parts[1])
            if file_end >= start_year and file_start <= end_year:
                input_files.append(f)

        if not input_files:
            raise FileNotFoundError(
                f"No ocean thermal forcing files for year range "
                f"{start_year}-{end_year}")

        logger.info(f"Found {len(input_files)} ocean thermal forcing files "
                    f"overlapping years {start_year}-{end_year}")

        # Build mapping file using the first input file as grid template.
        # Ocean grid (761x761, ~8km) differs from atmosphere (3041x3041, 2km).
        mapping_file = (f"map_ismip7_ocean_8km_to_{mali_mesh_name}_"
                        f"{method_remap}.nc")

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

            logger.info(f"  Remapping: {basename}")
            args = ["ncremap",
                    "-i", input_file,
                    "-o", remapped_file,
                    "-m", mapping_file,
                    "-v", "tf"]

            check_call(args, logger=logger)

        # Combine remapped files and rename to MALI conventions
        logger.info("Combining remapped files and renaming variables...")
        output_file = (f"{mali_mesh_name}_thermal_forcing_{model}_{scenario}_"
                       f"{start_year}-{end_year}.nc")

        self._combine_and_rename(remapped_files, output_file,
                                 start_year, end_year)

        # Clean up remapped files
        logger.info("Cleaning up temporary remapped files...")
        for f in remapped_files:
            if os.path.exists(f):
                os.remove(f)

        # Place output in appropriate directory
        output_path = os.path.join(output_base_path, "ocean_thermal_forcing",
                                   f"{model}_{scenario}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        dst = os.path.join(output_path, output_file)
        shutil.copy(output_file, dst)

        logger.info(f"Done. Output: {dst}")

    def _combine_and_rename(self, remapped_files, output_file,
                            start_year, end_year):
        """
        Combine decade-spanning remapped files, subset to the requested
        year range, and rename variables/dimensions to MALI conventions.

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
                               combine="nested", engine="netcdf4")

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
