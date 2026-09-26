import os
import shutil

import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call

from compass.landice.ismip7.archive import (
    mapping_file_name,
    resolve_ocean_source,
    resolve_version_directory,
)
from compass.landice.ismip7.ice_sheet_params import get_params
from compass.landice.ismip7.remap import extrapolate_source
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

        if section.getboolean('process_ocean_thermal'):
            params = get_params(config)
            if params.get('ocean_choice_layout', False):
                choice = self._get_ocean_choices(config)[0]
                source = resolve_ocean_source(config, choice=choice)
            else:
                source = resolve_ocean_source(config)
            method_remap = config.get(
                'ismip7_ocean_thermal', 'method_remap')
            mapping_file = mapping_file_name(
                config, 'ocean', source.source_grid, method_remap)
            self.add_input_file(
                filename=mapping_file,
                target=f'../build_mapping_file/{mapping_file}')

        if section.getboolean('process_ocean_climatology'):
            method_remap = config.get(
                'ismip7_ocean_climatology', 'method_remap')
            mapping_file = mapping_file_name(
                config, 'ocean', 'climatology', method_remap)
            self.add_input_file(
                filename=mapping_file,
                target=f'../build_mapping_file/{mapping_file}')

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
        (e.g., CESM2-WACCM historical or ssp585) or, for AIS OCX, from one
        or more reanalysis ocean "choices" (main/cold/warm/vary).
        """
        config = self.config
        params = get_params(config)

        section = config["ismip7"]
        mali_mesh_name = section.get("mali_mesh_name")
        scenario = section.get("scenario")
        output_base_path = section.get("output_base_path")

        section = config["ismip7_ocean_thermal"]
        method_remap = section.get("method_remap")
        start_year = section.getint("start_year")
        end_year = section.getint("end_year")

        ocean_3d = params['ocean_3d']

        # Assemble the list of forcing sources to process. AIS OCX has several
        # ocean "choices" (main/cold/warm/vary) in per-choice subdirectories
        # with no model token in the filename; all other cases (ESM scenarios
        # and GrIS OCX) have a single source.
        jobs = []
        if params.get('ocean_choice_layout', False):
            for choice in self._get_ocean_choices(config):
                jobs.append(resolve_ocean_source(config, choice=choice))
        else:
            jobs.append(resolve_ocean_source(config))

        for job in jobs:
            self.logger.info(f"Using ocean source {job.directory}")
            mapping_file = mapping_file_name(
                config, "ocean", job.source_grid, method_remap)
            self._process_ocean_forcing(
                job, mapping_file, ocean_3d, start_year, end_year,
                mali_mesh_name, output_base_path)

        # AIS OCX writes one ocean_thermal_forcing dir per ocean choice
        # (OCX_<choice>), but atmosphere output is written once, under OCX.
        # Mirror the atmosphere files into each OCX_<choice> directory so
        # ismip7_run can treat it as a single, self-contained forcing dir.
        if params.get('ocean_choice_layout', False):
            for job in jobs:
                self._link_atmosphere_outputs(
                    output_base_path, scenario, job.forcing_group)

    def _get_ocean_choices(self, config):
        """
        Parse the comma-separated ``ocean_choice`` option (AIS OCX only) into a
        list of ocean choices, expanding ``all`` to every available choice.

        Parameters
        ----------
        config : compass.config.CompassConfigParser
            Configuration options for the test case

        Returns
        -------
        choices : list of str
            Ordered, de-duplicated list of ocean choices to process
        """
        valid = ["main", "cold", "warm", "vary"]
        raw = config.get("ismip7_ocean_thermal", "ocean_choice")
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        if any(t.lower() == "all" for t in tokens):
            return valid

        choices = []
        for token in tokens:
            if token not in valid:
                raise ValueError(
                    f"Invalid ocean_choice '{token}'. Must be a "
                    f"comma-separated list of {valid} or 'all'.")
            if token not in choices:
                choices.append(token)
        if not choices:
            raise ValueError(
                "No ocean_choice specified for AIS OCX ocean forcing.")
        return choices

    def _link_atmosphere_outputs(self, output_base_path, atm_forcing_group,
                                 choice_forcing_group):
        """
        Symlink the shared atmosphere output files into an ocean-choice
        forcing directory (e.g. OCX_main) so it mirrors the atmosphere's
        directory (e.g. OCX) and can be used standalone as an
        ``ocx_forcing_path`` by ismip7_run.

        Parameters
        ----------
        output_base_path : str
            Base path under which output is written

        atm_forcing_group : str
            Name of the directory under ``output_base_path`` that holds the
            actual atmosphere output files (e.g. 'OCX')

        choice_forcing_group : str
            Name of the ocean-choice directory to populate with symlinks
            (e.g. 'OCX_main')
        """
        logger = self.logger
        src_dir = os.path.join(output_base_path, atm_forcing_group,
                               "atmosphere")
        if not os.path.isdir(src_dir):
            logger.warning(
                f"Atmosphere output not found at {src_dir}; skipping "
                f"atmosphere symlinks for {choice_forcing_group}. Run the "
                f"atmosphere test case to populate it.")
            return

        dst_dir = os.path.join(output_base_path, choice_forcing_group,
                               "atmosphere")

        if os.path.realpath(dst_dir) == os.path.realpath(src_dir):
            logger.warning(
                f"Skipping atmosphere mirror for {choice_forcing_group}: "
                f"destination {dst_dir} resolves to the source directory.")
            return

        os.makedirs(dst_dir, exist_ok=True)

        for fname in os.listdir(src_dir):
            src = os.path.join(src_dir, fname)
            if os.path.islink(src) or not os.path.isfile(src):
                continue
            dst = os.path.join(dst_dir, fname)
            if os.path.realpath(dst) == os.path.realpath(src):
                logger.warning(
                    f"Skipping {fname}: destination resolves to source.")
                continue
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(src, dst)
            logger.info(f"  Linked {dst} -> {src}")

    def _process_ocean_forcing(self, job, mapping_file, ocean_3d,
                               start_year, end_year, mali_mesh_name,
                               output_base_path):
        """
        Discover, remap, combine, and save the thermal forcing for a single
        forcing source described by ``job``.

        Parameters
        ----------
        job : compass.landice.ismip7.archive.ForcingSource
            Resolved ocean forcing source
        mapping_file : str
            Path of the shared ocean-to-MALI mapping file
        ocean_3d : bool
            Whether the thermal forcing is 3D (AIS) or 2D (GrIS)
        start_year, end_year : int
            Inclusive year range to process
        mali_mesh_name : str
            MALI mesh name
        output_base_path : str
            Base path under which output is written
        """
        logger = self.logger
        forcing_group = job.forcing_group
        label = job.label
        all_files = job.files

        # Filter to files that overlap with the requested year range.
        # AIS files are named with decade or multi-decade ranges (e.g.,
        # 1850-1859, 1950-2025). GrIS files are named with single years.
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

        # Remap each file
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
                extrapolate_source(input_file, extrap_file, "tf",
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
        tf_label = "3dThermalForcing" if ocean_3d else "2dThermalForcing"
        output_file = (f"{mali_mesh_name}_{tf_label}_{label}_"
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
        output_base_path = section.get("output_base_path")

        section = config["ismip7_ocean_climatology"]
        method_remap = section.get("method_remap")
        base_path_climatology = section.get("base_path_climatology")
        requested_version = section.get("version", fallback="latest")

        # Discover climatology TF file
        _, _, all_files = resolve_version_directory(
            os.path.join(base_path_climatology, "tf"), requested_version,
            "tf_*.nc")

        # Use the first (and likely only) file
        input_file = all_files[0]
        logger.info(f"Processing ocean TF climatology: "
                    f"{os.path.basename(input_file)}")

        # The mapping file is supplied by the build_mapping_file step.
        mapping_file = mapping_file_name(
            config, "ocean", "climatology", method_remap)

        # Extrapolate and remap
        basename = os.path.basename(input_file)
        remapped_file = f"remapped_{basename}"

        if not os.path.exists(remapped_file):
            extrap_file = f"extrap_{basename}"
            if not os.path.exists(extrap_file):
                extrapolate_source(input_file, extrap_file, "tf",
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
        version = os.path.basename(os.path.dirname(input_file))
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
