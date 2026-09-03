import glob
import os
import shutil

import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call

from compass.landice.tests.ismip7_forcing.create_mapfile import (
    build_mapping_file,
)
from compass.landice.tests.ismip7_forcing.ice_sheet_params import get_params
from compass.step import Step


class ProcessTemperature(Step):
    """
    A step for processing ISMIP7 ice surface temperature (ts) data.
    Remaps monthly temperature from the ISMIP7 2km polar stereographic
    grid to the MALI unstructured mesh.
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_forcing.atmosphere.Atmosphere
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name="process_temperature")

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
        params = get_params(config)

        section = config["ismip7"]
        base_path_ismip7 = section.get("base_path_ismip7")
        mali_mesh_name = section.get("mali_mesh_name")
        mali_mesh_file = section.get("mali_mesh_file")
        model = section.get("model")
        scenario = section.get("scenario")
        output_base_path = section.get("output_base_path")

        section = config["ismip7_atmosphere"]
        method_remap = section.get("method_remap")
        start_year = section.getint("start_year")
        end_year = section.getint("end_year")

        # Discover input files
        prefix = params['prefix']
        resolution = params['atm_resolution']
        version = params['atm_version']
        if params['atm_model'] is not None:
            model = params['atm_model']
        input_path = os.path.join(base_path_ismip7, "ts", version)
        file_pattern = (f"ts_{prefix}_{model}_{scenario}_"
                        f"SDBN1-{resolution}_{version}_*.nc")
        all_files = sorted(glob.glob(os.path.join(input_path, file_pattern)))

        if not all_files:
            raise FileNotFoundError(
                f"No temperature files found matching pattern:\n"
                f"  {os.path.join(input_path, file_pattern)}")

        # Filter to requested year range
        input_files = []
        for f in all_files:
            # skip non-yearly files such as climatology averages (*_avg.nc)
            token = os.path.basename(f).split("_")[-1].replace(".nc", "")
            if not token.isdigit():
                continue
            year = int(token)
            if start_year <= year <= end_year:
                input_files.append(f)

        if not input_files:
            raise FileNotFoundError(
                f"No temperature files for year range "
                f"{start_year}-{end_year}")

        logger.info(f"Found {len(input_files)} temperature files for years "
                    f"{start_year}-{end_year}")

        # Build mapping file (reuse if already created by process_smb)
        ice_sheet = config.get("ismip7", "ice_sheet")
        mapping_file = (f"map_ismip7_{ice_sheet}_atm_to_"
                        f"{mali_mesh_name}_{method_remap}.nc")

        if not os.path.exists(mapping_file):
            logger.info("Building mapping file...")
            build_mapping_file(config, logger,
                               input_files[0], mapping_file,
                               mali_mesh_file=mali_mesh_file,
                               method_remap=method_remap)

        # Remap each year file
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
                    "-v", "ts"]

            check_call(args, logger=logger)

        # Combine remapped files and rename to MALI conventions
        logger.info("Combining remapped files and renaming variables...")
        output_file = (f"{mali_mesh_name}_temperature_{model}_{scenario}_"
                       f"{start_year}-{end_year}.nc")

        self._combine_and_rename(remapped_files, output_file)

        # Clean up remapped files
        logger.info("Cleaning up temporary remapped files...")
        for f in remapped_files:
            if os.path.exists(f):
                os.remove(f)

        # Place output in appropriate directory
        output_path = os.path.join(output_base_path, "atmosphere_forcing",
                                   f"{model}_{scenario}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        dst = os.path.join(output_path, output_file)
        shutil.copy(output_file, dst)

        logger.info(f"Done. Output: {dst}")

    def _combine_and_rename(self, remapped_files, output_file):
        """
        Combine yearly remapped files and rename variables/dimensions
        to MALI conventions.

        Parameters
        ----------
        remapped_files : list of str
            List of remapped NetCDF file paths

        output_file : str
            Output file path
        """
        ds = xr.open_mfdataset(remapped_files, concat_dim="time",
                               combine="nested", engine="netcdf4",
                               drop_variables="time_bnds")

        # Rename dimensions to MALI conventions
        rename_dims = {}
        if "time" in ds.dims:
            rename_dims["time"] = "Time"
        if "ncol" in ds.dims:
            rename_dims["ncol"] = "nCells"
        if rename_dims:
            ds = ds.rename(rename_dims)

        # Rename variable
        if "ts" in ds:
            ds = ds.rename({"ts": "surfaceAirTemperature"})

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
        ds["surfaceAirTemperature"].attrs = {
            "long_name": "temperature at top of ice sheet model",
            "units": "K",
        }

        # Drop auxiliary variables from remapping
        vars_to_drop = [v for v in ["lon", "lon_vertices", "lat",
                                    "lat_vertices", "area"]
                        if v in ds]
        if vars_to_drop:
            ds = ds.drop_vars(vars_to_drop)

        # Drop Time coordinate values (keep as dimension only);
        # MALI uses xtime, not CF-encoded time coordinates
        if "Time" in ds.coords:
            ds = ds.drop_vars("Time")

        write_netcdf(ds, output_file)
        ds.close()
