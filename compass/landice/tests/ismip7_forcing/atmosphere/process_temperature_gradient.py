import os
import shutil

import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call

from compass.landice.ismip7.archive import (
    mapping_file_name,
    resolve_atmosphere_source,
)
from compass.landice.ismip7.remap import extrapolate_source
from compass.step import Step


class ProcessTemperatureGradient(Step):
    """
    A step for processing ISMIP7 temperature elevation gradient (dtsdz) data.
    Remaps the annual temperature gradient from the ISMIP7 2km polar
    stereographic grid to the MALI unstructured mesh. This field is used
    for temperature-elevation feedback corrections.
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_forcing.atmosphere.Atmosphere
            The test case this step belongs to
        """
        super().__init__(test_case=test_case,
                         name="process_temperature_gradient")

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

        method_remap = config.get('ismip7_atmosphere', 'method_remap')
        source = resolve_atmosphere_source(config, 'dtsdz')
        mapping_file = mapping_file_name(
            config, 'atm', source.source_grid, method_remap)
        self.add_input_file(
            filename=mapping_file,
            target=f'../build_mapping_file/{mapping_file}')

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        section = config["ismip7"]
        mali_mesh_name = section.get("mali_mesh_name")
        scenario = section.get("scenario")
        output_base_path = section.get("output_base_path")

        section = config["ismip7_atmosphere"]
        method_remap = section.get("method_remap")
        start_year = section.getint("start_year")
        end_year = section.getint("end_year")

        source = resolve_atmosphere_source(config, "dtsdz")
        all_files = source.files
        model = source.model
        forcing_group = source.forcing_group
        logger.info(f"Using atmosphere source {source.directory}")

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
                f"No temperature gradient files for year range "
                f"{start_year}-{end_year}")

        logger.info(f"Found {len(input_files)} temperature gradient files "
                    f"for years {start_year}-{end_year}")

        # The mapping file is supplied by the build_mapping_file step.
        mapping_file = mapping_file_name(
            config, "atm", source.source_grid, method_remap)

        # Remap each year file
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
                extrapolate_source(input_file, extrap_file, "dtsdz", logger)

            logger.info(f"  Remapping: {basename}")
            args = ["ncremap",
                    "-i", extrap_file,
                    "-o", remapped_file,
                    "-m", mapping_file,
                    "-v", "dtsdz"]

            check_call(args, logger=logger)

            # Clean up extrapolated source file
            os.remove(extrap_file)

        # Combine remapped files and rename to MALI conventions
        logger.info("Combining remapped files and renaming variables...")
        output_file = (f"{mali_mesh_name}_temperature_gradient_{model}_"
                       f"{scenario}_{start_year}-{end_year}.nc")

        self._combine_and_rename(remapped_files, output_file)

        # Clean up remapped files
        logger.info("Cleaning up temporary remapped files...")
        for f in remapped_files:
            if os.path.exists(f):
                os.remove(f)

        # Place output in appropriate directory
        output_path = os.path.join(output_base_path, forcing_group,
                                   "atmosphere")
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

        # Rename to MALI convention (PR #169)
        if "dtsdz" in ds:
            ds = ds.rename({"dtsdz": "surfaceAirTemperatureLapseRate"})

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
        ds["surfaceAirTemperatureLapseRate"].attrs = {
            "long_name": "vertical gradient dTdz used for SAT "
                         "elevation-change correction",
            "units": "K m-1",
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
