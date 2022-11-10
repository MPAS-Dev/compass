import os
import shutil
import xarray as xr
from compass.landice.tests.ismip6_forcing.create_mapfile \
    import build_mapping_file
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from compass.step import Step


class ProcessBasalMelt(Step):
    """
    A step for processing (combine, remap and rename) the ISMIP6 basalmelt
    data
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip6_forcing.ocean_basal.OceanBasal
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name="process_basal_melt",
                         ntasks=4, min_tasks=1)

    def setup(self):
        """
        Set up this step of the test case
        """
        config = self.config
        section = config["ismip6_ais"]
        base_path_ismip6 = section.get("base_path_ismip6")
        base_path_mali = section.get("base_path_mali")
        mali_mesh_name = section.get("mali_mesh_name")
        mali_mesh_file = section.get("mali_mesh_file")
        period_endyear = section.get("period_endyear")

        input_file_basin = self._file_basin[period_endyear]
        input_files_coeff = self._files_coeff[period_endyear]
        self.add_input_file(filename=mali_mesh_file,
                            target=os.path.join(base_path_mali,
                                                mali_mesh_file))
        self.add_input_file(filename=os.path.basename(input_file_basin),
                            target=os.path.join(base_path_ismip6,
                                                input_file_basin))

        for file in input_files_coeff:
            self.add_input_file(filename=os.path.basename(file),
                                target=os.path.join(base_path_ismip6, file))
            self.add_output_file(filename=f"{mali_mesh_name}_basin_and_"
                                          f"{os.path.basename(file)}")

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        section = config["ismip6_ais"]
        mali_mesh_name = section.get("mali_mesh_name")
        mali_mesh_file = section.get("mali_mesh_file")
        period_endyear = section.get("period_endyear")
        output_base_path = section.get("output_base_path")
        # we always want neareststod for the remapping method because we want
        # a single value per basin
        method_remap = "neareststod"

        # combine, interpolate and rename the basin file and deltaT0_gamma0
        # ismip6 input files

        # call the function that combines data
        logger.info(f"Combining the ismip6 input files")
        input_file_basin = self._file_basin[period_endyear]
        input_file_list = self._files_coeff[period_endyear]

        for i, file in enumerate(input_file_list):
            logger.info(f"!---Start processing the current file---!")
            logger.info(f"processing the input file "
                        f"'{os.path.basename(file)}'")
            # temporary file names. Note: The counter 'i' seems to be necessary
            # for 'ncremap' to correctly interpolate the gamma0 values for an
            # unknown reason.
            combined_file_temp = f"combined_{i}.nc"
            remapped_file_temp = f"remapped_{i}.nc"

            self.combine_ismip6_inputfiles(os.path.basename(input_file_basin),
                                           os.path.basename(file),
                                           combined_file_temp)

            # remap the input forcing file.
            logger.info(f"Calling the remapping function...")
            self.remap_ismip6_basal_melt_to_mali_vars(combined_file_temp,
                                                      remapped_file_temp,
                                                      mali_mesh_name,
                                                      mali_mesh_file,
                                                      method_remap)

            output_file = f"{mali_mesh_name}_basin_and_" \
                          f"{os.path.basename(file)}"

            # rename the ismip6 variables to MALI variables
            logger.info(f"Renaming the ismip6 variables to "
                        f"mali variable names...")
            self.rename_ismip6_basal_melt_to_mali_vars(remapped_file_temp,
                                                       output_file)

            logger.info(f"Remapping and renamping process done successfully. "
                        f"Removing the temporary files...")

            # remove the temporary combined file
            os.remove(combined_file_temp)
            os.remove(remapped_file_temp)

            # place the output file in appropriate director
            output_path = f"{output_base_path}/basal_melt/parameterizations/"
            if not os.path.exists(output_path):
                logger.info("Creating a new directory for the output data...")
                os.makedirs(output_path)

            src = os.path.join(os.getcwd(), output_file)
            dst = os.path.join(output_path, output_file)
            shutil.copy(src, dst)

            logger.info(f"!---Done processing the current file---!")
            logger.info(f"")
            logger.info(f"")

    def combine_ismip6_inputfiles(self, basin_file, coeff_gamma0_deltat_file,
                                  combined_file_temp):
        """
        Combine ismip6 input files before regridding onto mali mesh

        Parameters
        ----------
        basin_file : str
            imbie2 basin numbers in ismip6 grid

        coeff_gamma0_deltat_file : str
            uniform melt rate coefficient (gamma0) and temperature
            correction per basin

        combined_file_temp : str
            temporary output file that has all the variables combined
        """

        ds_basin = xr.open_dataset(basin_file, engine="netcdf4")
        ds = xr.open_dataset(coeff_gamma0_deltat_file, engine="netcdf4")

        ds["ismip6shelfMelt_basin"] = ds_basin.basinNumber
        write_netcdf(ds, combined_file_temp)

    def remap_ismip6_basal_melt_to_mali_vars(self, input_file, output_file,
                                             mali_mesh_name, mali_mesh_file,
                                             method_remap):
        """
        Remap the input ismip6 basal melt data onto mali mesh

        Parameters
        ----------
        input_file: str
            temporary output file that has all the variables combined
            combined_file_temp generated in the above function

        output_file : str
            ismip6 data remapped on mali mesh

        mali_mesh_name : str
            name of the mali mesh used to name mapping files

        mali_mesh_file : str, optional
            The MALI mesh file if mapping file does not exist

        method_remap : str, optional
            Remapping method used in building a mapping file
        """

        # check if mapfile exists
        mapping_file = f"map_ismip6_8km_to_{mali_mesh_name}_{method_remap}.nc"

        if not os.path.exists(mapping_file):
            # build a mapping file if it doesn't already exist
            build_mapping_file(self.config, self.ntasks, self.logger,
                               input_file, mapping_file, mali_mesh_file,
                               method_remap)
        else:
            self.logger.info(f"Mapping file exists. "
                             f"Remapping the input data...")

        # remap the input data
        args = ["ncremap",
                "-i", input_file,
                "-o", output_file,
                "-m", mapping_file]

        check_call(args, logger=self.logger)

    def rename_ismip6_basal_melt_to_mali_vars(self, remapped_file_temp,
                                              output_file):
        """
        Rename variables in the remapped ismip6 input data
        to the ones that MALI uses.

        Parameters
        ----------
        remapped_file_temp : str
            temporary ismip6 data remapped on mali mesh

        output_file : str
            remapped ismip6 data renamed on mali mesh
        """

        # open dataset in 20 years chunk
        ds = xr.open_dataset(remapped_file_temp, chunks=dict(time=20))

        # build dictionary and rename the ismip6 dimension and variables
        ismip6_to_mali_dims = dict(
            ncol="nCells")
        ds = ds.rename(ismip6_to_mali_dims)

        ismip6_to_mali_vars = dict(
            deltaT_basin="ismip6shelfMelt_deltaT",
            gamma0="ismip6shelfMelt_gamma0")

        # drop unnecessary variables on the regridded data on MALI mesh
        ds = ds.rename(ismip6_to_mali_vars)
        ds = ds.drop_vars(["lat_vertices", "lon_vertices",
                           "lat", "lon", "area"])

        # write to a new netCDF file
        write_netcdf(ds, output_file)
        ds.close()

    # input files: input uniform melt rate coefficient (gamma0)
    # and temperature correction per basin
    _file_basin = {
        "2100": "AIS/Ocean_Forcing/imbie2/imbie2_basin_numbers_8km.nc",
        "2300": "AIS/Ocean_forcing/imbie2/imbie2_basin_numbers_8km.nc"
    }

    _files_coeff = {
        "2100": ["AIS/Ocean_Forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_local_5th_pct_PIGL_gamma_calibration.nc",
                 "AIS/Ocean_Forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_local_5th_percentile.nc",
                 "AIS/Ocean_Forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_local_95th_pct_PIGL_gamma_calibration.nc",
                 "AIS/Ocean_Forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_local_95th_percentile.nc",
                 "AIS/Ocean_Forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_local_median_PIGL_gamma_calibration.nc",
                 "AIS/Ocean_Forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_local_median.nc",
                 "AIS/Ocean_Forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_non_local_5th_pct_PIGL_gamma_calibration.nc",
                 "AIS/Ocean_Forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_non_local_5th_percentile.nc",
                 "AIS/Ocean_Forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_non_local_95th_pct_PIGL_gamma_calibration.nc",
                 "AIS/Ocean_Forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_non_local_95th_percentile.nc",
                 "AIS/Ocean_Forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_non_local_median_PIGL_gamma_calibration.nc",
                 "AIS/Ocean_Forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_non_local_median.nc"],

        "2300": ["AIS/Ocean_forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_local_5th_pct_PIGL_gamma_calibration.nc",
                 "AIS/Ocean_forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_local_5th_percentile.nc",
                 "AIS/Ocean_forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_local_95th_pct_PIGL_gamma_calibration.nc",
                 "AIS/Ocean_forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_local_95th_percentile.nc",
                 "AIS/Ocean_forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_local_median_PIGL_gamma_calibration.nc",
                 "AIS/Ocean_forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_local_median.nc",
                 "AIS/Ocean_forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_non_local_5th_pct_PIGL_gamma_calibration.nc",
                 "AIS/Ocean_forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_non_local_5th_percentile.nc",
                 "AIS/Ocean_forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_non_local_95th_pct_PIGL_gamma_calibration.nc",
                 "AIS/Ocean_forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_non_local_95th_percentile.nc",
                 "AIS/Ocean_forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_non_local_median_PIGL_gamma_calibration.nc",
                 "AIS/Ocean_forcing/parameterizations/coeff_gamma0_DeltaT_quadratic_non_local_median.nc"]
    }