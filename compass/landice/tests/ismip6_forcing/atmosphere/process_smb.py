import os
import numpy as np
import shutil
import xarray as xr
from compass.landice.tests.ismip6_forcing.atmosphere.create_mapfile_smb \
    import build_mapping_file
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from compass.step import Step


class ProcessSMB(Step):
    """
    A step for processing the ISMIP6 surface mass balance data
    """

    def __init__(self, test_case, input_file=None):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip6_forcing.atmosphere.Atmosphere
            The test case this step belongs to

        input_file : file name of ismip6 forcing data processed by this step
        """
        self.input_file = input_file
        super().__init__(test_case=test_case, name="process_smb", ntasks=4,
                         min_tasks=1)

    def setup(self):
        """
        Set up this step of the test case
        """
        config = self.config
        section = config["ismip6_ais"]
        base_path_ismip6 = section.get("base_path_ismip6")
        base_path_mali = section.get("base_path_mali")
        output_base_path = section.get("output_base_path")
        mali_mesh_name = section.get("mali_mesh_name")
        mali_mesh_file = section.get("mali_mesh_file")
        period_endyear = section.get("period_endyear")
        model = section.get("model")
        scenario = section.get("scenario")
        res_ismip6 = section.get("res_ismip6")

        self.add_input_file(filename=mali_mesh_file,
                            target=os.path.join(base_path_mali,
                                                mali_mesh_file))

        input_file_list = \
            self._files[period_endyear][model][scenario][res_ismip6]

        for file in input_file_list:
            self.add_input_file(filename=os.path.basename(file),
                                target=os.path.join(base_path_ismip6,
                                                    file))

        output_file_esm = f"{mali_mesh_name}_SMB_{model}_{scenario}_" \
                          f"{period_endyear}.nc"
        self.add_output_file(filename=output_file_esm)

        # add processed racmo data as input as it is needed for smb correction
        racmo_clim_file = f"{mali_mesh_name}_RACMO2.3p2_ANT27" \
                          f"_smb_climatology_1995-2017.nc"
        racmo_path = f"{output_base_path}/atmosphere_forcing/" \
                     f"RACMO_climatology_1995-2017"

        self.add_input_file(filename=racmo_clim_file,
                            target=os.path.join(racmo_path,
                                                racmo_clim_file))

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
        model = section.get("model")
        scenario = section.get("scenario")
        res_ismip6 = section.get("res_ismip6")
        output_base_path = section.get("output_base_path")

        section = config["ismip6_ais_atmosphere"]
        method_remap = section.get("method_remap")

        # define file names needed
        # input racmo climotology file
        racmo_clim_file = f"{mali_mesh_name}_RACMO2.3p2_ANT27" \
                          f"_smb_climatology_1995-2017.nc"
        racmo_path = f"{output_base_path}/atmosphere_forcing/" \
                     f"RACMO_climatology_1995-2017"
        # check if the processed racmo input file exists
        racmo_clim_file_final = os.path.join(racmo_path, racmo_clim_file)
        if not os.path.exists(racmo_clim_file_final):
            raise ValueError("Processed RACMO data does not exist, "
                             "but it is required as an input file "
                             "to run this step. Please run `process_smb_racmo`"
                             "step prior to running this step by setting"
                             "the config option 'process_smb_racmo' to 'True'.")

        # temporary remapped climatology and anomaly files
        clim_ismip6_temp = "clim_ismip6.nc"
        remapped_clim_ismip6_temp = "remapped_clim_ismip6.nc"
        remapped_anomaly_ismip6_temp = "remapped_anomaly_ismip6.nc"
        # renamed remapped climatology and anomaly files (final outputs)
        output_clim_ismip6 = f"{mali_mesh_name}_SMB_climatology_1995-2017_" \
                             f"{model}_{scenario}.nc"
        output_anomaly_ismip6 = f"{mali_mesh_name}_SMB_{model}_{scenario}_" \
                                f"{period_endyear}.nc"

        # combine ismip6 forcing data covering different periods
        # into a single file
        input_file_list = \
            self._files[period_endyear][model][scenario][res_ismip6]

        i = 0
        for file in input_file_list:
            input_file_list[i] = os.path.basename(file)
            i += 1

        input_file_combined = xr.open_mfdataset(input_file_list,
                                                concat_dim='time',
                                                combine='nested',
                                                engine='netcdf4')
        combined_file_temp = "combined.nc"
        write_netcdf(input_file_combined, combined_file_temp)

        # create smb climatology data over 1995-2017
        # take the time average over the period 1995-2017
        # note: make sure to have the correct time indexing for each
        # smb anomaly files on which climatology is calculated.
        logger.info(f"Calculating climatology for {model}_{scenario} forcing"
                    f"over 1995-2017")
        args = [f"ncra", "-O", "-F", "-d", "time,1,23",
                f"{combined_file_temp}",
                f"{clim_ismip6_temp}"]

        check_call(args, logger=logger)

        # remap and rename the ismip6 smb climatology
        logger.info("Remapping ismip6 climatology onto MALI mesh...")
        self.remap_ismip6_smb_to_mali(clim_ismip6_temp,
                                      remapped_clim_ismip6_temp,
                                      mali_mesh_name,
                                      mali_mesh_file,
                                      method_remap)

        # rename the ismip6 variables to MALI variables
        logger.info("Renaming the ismip6 variables to mali variable names...")
        self.rename_ismip6_smb_to_mali_vars(remapped_clim_ismip6_temp,
                                            output_clim_ismip6)

        # remap and rename ismip6 smb anomaly
        logger.info(f"Remapping the {model}_{scenario} SMB anomaly onto "
                    f"MALI mesh")
        self.remap_ismip6_smb_to_mali(combined_file_temp,
                                      remapped_anomaly_ismip6_temp,
                                      mali_mesh_name,
                                      mali_mesh_file,
                                      method_remap)

        # rename the ismip6 variables to MALI variables
        logger.info("Renaming the ismip6 variables to mali variable names...")
        self.rename_ismip6_smb_to_mali_vars(remapped_anomaly_ismip6_temp,
                                            output_anomaly_ismip6)

        # correct the SMB anomaly field with mali base SMB field
        logger.info("Correcting the SMB anomaly field for the base SMB "
                    "climatology 1995-2017...")
        self.correct_smb_anomaly_for_climatology(racmo_clim_file,
                                                 output_clim_ismip6,
                                                 output_anomaly_ismip6)

        logger.info("Processing done successfully. "
                    "Removing the temporary files...")
        # remove the temporary remapped and combined files
        os.remove(remapped_clim_ismip6_temp)
        os.remove(remapped_anomaly_ismip6_temp)
        os.remove(combined_file_temp)
        os.remove(clim_ismip6_temp)
        os.remove(output_clim_ismip6)

        # place the output file in appropriate directory
        output_path = f"{output_base_path}/atmosphere_forcing/" \
                      f"{model}_{scenario}/1995-{period_endyear}"
        if not os.path.exists(output_path):
            print("Creating a new directory for the output data...")
            os.makedirs(output_path)

        src = os.path.join(os.getcwd(), output_anomaly_ismip6)
        dst = os.path.join(output_path, output_anomaly_ismip6)
        shutil.copy(src, dst)

        logger.info(f"!---Done processing the file---!")

    def remap_ismip6_smb_to_mali(self, input_file, output_file, mali_mesh_name,
                                 mali_mesh_file, method_remap):
        """
        Remap the input ismip6 smb forcing data onto mali mesh

        Parameters
        ----------
        input_file: str
            input smb data on its native polarstereo 8km grid

        output_file : str
            smb data remapped on mali mesh

        mali_mesh_name : str
            name of the mali mesh used to name mapping files

        mali_mesh_file : str, optional
            The MALI mesh file if mapping file does not exist

        method_remap : str, optional
            Remapping method used in building a mapping file
        """
        mapping_file = f"map_ismip6_8km_to_" \
                       f"{mali_mesh_name}_{method_remap}.nc"

        # check if mapfile exists
        if not os.path.exists(mapping_file):
            # build a mapping file if it doesn't already exist
            self.logger.info(f"Creating a mapping file. "
                             f"Mapping method used: {method_remap}")
            build_mapping_file(self.config, self.ntasks, self.logger,
                               input_file, mapping_file, mali_mesh_file,
                               method_remap)
        else:
            self.logger.info("Mapping file exists. "
                             "Remapping the input data...")

        # remap the input data
        args = ["ncremap",
                "-i", input_file,
                "-o", output_file,
                "-m", mapping_file,
                "-v", "smb_anomaly"]

        check_call(args, logger=self.logger)

    def rename_ismip6_smb_to_mali_vars(self, remapped_file_temp, output_file):
        """
        Rename variables in the remapped source input data
        to the ones that MALI uses.

        Parameters
        ----------
        remapped_file_temp : str
            temporary ismip6 data remapped on mali mesh

        output_file : str
            remapped ismip6 data renamed on mali mesh
        """

        # open dataset in 20 years chunk
        ds = xr.open_dataset(remapped_file_temp, chunks=dict(time=20),
                             engine="netcdf4")

        # build dictionary for ismip6 variables that MALI takes in
        ismip6_to_mali_dims = dict(
            time="Time",
            ncol="nCells")
        ds = ds.rename(ismip6_to_mali_dims)

        ismip6_to_mali_vars = dict(
            smb_anomaly="sfcMassBal")
        ds = ds.rename(ismip6_to_mali_vars)

        # add xtime variable
        xtime = []
        for t_index in range(ds.sizes["Time"]):
            date = ds.Time[t_index]
            # forcing files do not all match even years, so round up the years
            # pandas round function does not work for years, so do it manually
            yr = date.dt.year.values
            mo = date.dt.month.values
            dy = date.dt.day.values
            dec_yr = np.around(yr + (30 * (mo - 1) + dy) / 365.0)
            date = f"{dec_yr.astype(int)}-01-01_00:00:00".ljust(64)
            xtime.append(date)

        ds["xtime"] = ("Time", xtime)
        ds["xtime"] = ds.xtime.astype('S')

        # drop unnecessary variables
        ds = ds.drop_vars(["lon", "lon_vertices", "lat", "lat_vertices",
                           "area"])

        # write to a new netCDF file
        write_netcdf(ds, output_file)
        ds.close()

    def correct_smb_anomaly_for_climatology(self,
                                            racmo_clim_file,
                                            output_clim_ismip6_file,
                                            output_file_final):

        """
        Apply the MALI base SMB to the ismip6 SMB anomaly field

        Parameters
        ----------
        racmo_clim_file : str
            RACMO climatology file (1995-2017)

        output_clim_ismip6_file : str
            remapped and renamed ismip6 climatology file

        output_file_final : str
             climatology-corrected, final ismip6 smb anomaly file
        """

        ds = xr.open_dataset(output_file_final)
        ds_racmo_clim = xr.open_dataset(racmo_clim_file)
        ds_ismip6_clim = xr.open_dataset(output_clim_ismip6_file)

        # calculate the climatology correction
        corr_clim = (ds_racmo_clim["sfcMassBal"].isel(Time=0) -
                     ds_ismip6_clim["sfcMassBal"].isel(Time=0))

        # correct the ismip6 smb anomaly
        ds["sfcMassBal"] = ds["sfcMassBal"] + corr_clim

        # write metadata
        ds["sfcMassBal"].attrs = {"long_name" : "surface mass balance",
                                  "units" : "kg m-2 s-1",
                                  "coordinates" : "lat lon"}

        # write to a new netCDF file
        write_netcdf(ds, output_file_final)
        ds.close()

    # create a nested dictionary for the ISMIP6 original forcing files including relative path
    # Note: these files needed to be explicitly listed because of inconsistencies that are
    # present in file naming conventions in the ISMIP6 source dataset.
    _files = {
        "2100": {
            "CCSM4": {
                "RCP85": [
                    "AIS/Atmosphere_Forcing/ccsm4_rcp8.5/Regridded_8km/CCSM4_8km_anomaly_1995-2100.nc"]
            },
            "CESM2": {
                "SSP585v1": [
                    "AIS/Atmosphere_Forcing/CESM2_ssp585/Regridded_8km/CESM2_anomaly_ssp585_1995-2100_8km_v1.nc"],
                "SSP585v2": [
                    "AIS/Atmosphere_Forcing/CESM2_ssp585/Regridded_8km/CESM2_anomaly_ssp585_1995-2100_8km_v2.nc"]
            },
            "CNRM_CM6": {
                "SSP126": [
                    "AIS/Atmosphere_Forcing/CNRM_CM6_ssp126/Regridded_8km/CNRM-CM6-1_anomaly_ssp126_1995-2100_8km_ISMIP6.nc"],
                "SSP585": [
                    "AIS/Atmosphere_Forcing/CNRM_CM6_ssp585/Regridded_8km/CNRM-CM6-1_anomaly_ssp585_1995-2100_8km_ISMIP6.nc"]
            },
            "CNRM_ESM2": {
                "SSP585": [
                    "AIS/Atmosphere_Forcing/CNRM_ESM2_ssp585/Regridded_8km/CNRM-ESM2-1_anomaly_ssp585_1995-2100_8km_ISMIP6.nc"]
            },
            "CSIRO-Mk3-6-0": {
                "RCP85": [
                    "AIS/Atmosphere_Forcing/CSIRO-Mk3-6-0_rcp85/Regridded_8km/CSIRO-Mk3-6-0_8km_anomaly_rcp85_1995-2100.nc"]
            },
            "HadGEM2-ES": {
                "RCP85": [
                    "AIS/Atmosphere_Forcing/HadGEM2-ES_rcp85/Regridded_8km/HadGEM2-ES_8km_anomaly_rcp85_1995-2100.nc"]
            },
            "IPSL-CM5A-MR": {
                "RCP26": [
                    "AIS/Atmosphere_Forcing/IPSL-CM5A-MR_rcp26/Regridded_8km/IPSL-CM5A-MR_8km_anomaly_rcp26_1995-2100.nc"],
                "RCP85": [
                    "AIS/Atmosphere_Forcing/IPSL-CM5A-MR_rcp85/Regridded_8km/IPSL-CM5A-MR_8km_anomaly_rcp85_1995-2100.nc"]
            },
            "MIROC-ESM-CHEM": {
                "RCP85": [
                    "AIS/Atmosphere_Forcing/miroc-esm-chem_rcp8.5/Regridded_8km/MIROC-ESM-CHEM_8km_anomaly_1995-2100.nc"]
            },
            "NorESM1-M": {
                "RCP26": [
                    "AIS/Atmosphere_Forcing/noresm1-m_rcp2.6/Regridded_8km/NorESM-M_8km_anomaly_rcp26_1995-2100.nc"],
                "RCP85": [
                    "AIS/Atmosphere_Forcing/noresm1-m_rcp8.5/Regridded_8km/NorESM-M_8km_anomaly_1995-2100.nc"]
            },
            "UKESM1-0-LL": {
                "SSP585": [
                    "AIS/Atmosphere_Forcing/UKESM1-0-LL/Regridded_8km/UKESM1-0-LL_anomaly_ssp585_1995-2100_8km.nc"]
            }
        },
        "2300": {
            "CCSM4": {
                "RCP85": {
                    "4km": [
                        "AIS/Atmospheric_forcing/CCSM4_RCP85/Regridded_04km/CCSM4_4km_anomaly_1995-2100.nc",
                        "AIS/Atmospheric_forcing/CCSM4_RCP85/Regridded_04km/CCSM4_4km_anomaly_2101-2300.nc"],
                    "8km": [
                        "AIS/Atmospheric_forcing/CCSM4_RCP85/Regridded_08km/CCSM4_8km_anomaly_1995-2100.nc",
                        "AIS/Atmospheric_forcing/CCSM4_RCP85/Regridded_08km/CCSM4_8km_anomaly_2101-2300.nc"]
                },
            },
            "CESM2-WACCM": {
                "SSP585": {
                    "4km": [
                        "AIS/Atmospheric_forcing/CESM2-WACCM_ssp585/Regridded_4km/CESM2-WACCM_4km_anomaly_ssp585_1995-2100.nc",
                        "AIS/Atmospheric_forcing/CESM2-WACCM_ssp585/Regridded_4km/CESM2-WACCM_4km_anomaly_ssp585_2101-2299.nc"],
                    "8km": [
                        "AIS/Atmospheric_forcing/CESM2-WACCM_ssp585/Regridded_8km/CESM2-WACCM_8km_anomaly_ssp585_1995-2100.nc",
                        "AIS/Atmospheric_forcing/CESM2-WACCM_ssp585/Regridded_8km/CESM2-WACCM_8km_anomaly_ssp585_2101-2299.nc"]
                },
                "SSP585-repeat": {
                    "4km": [
                        "AIS/Atmospheric_forcing/CESM2-WACCM_ssp585-repeat/Regridded_4km/CESM2-WACCM_4km_anomaly_ssp585_1995-2300_v2.nc"],
                    "8km": [
                        "AIS/Atmospheric_forcing/CESM2-WACCM_ssp585-repeat/Regridded_8km/CESM2-WACCM_8km_anomaly_ssp585_1995-2300_v2.nc"]
                },
            },
            "HadGEM2-ES": {
                "RCP85": {
                    "4km": [
                        "AIS/Atmospheric_forcing/HadGEM2-ES_RCP85/Regridded_4km/HadGEM2-ES_4km_anomaly_rcp85_1995-2100.nc",
                        "AIS/Atmospheric_forcing/HadGEM2-ES_RCP85/Regridded_4km/HadGEM2-ES_4km_anomaly_rcp85_2101-2299.nc"],
                    "8km": [
                        "AIS/Atmospheric_forcing/HadGEM2-ES_RCP85/Regridded_8km/HadGEM2-ES_8km_anomaly_rcp85_1995-2100.nc",
                        "AIS/Atmospheric_forcing/HadGEM2-ES_RCP85/Regridded_8km/HadGEM2-ES_8km_anomaly_rcp85_2101-2299.nc"]
                },
                "RCP85-repeat": {
                    "4km": [
                        "AIS/Atmospheric_forcing/HadGEM2-ES-RCP85-repeat/Regridded_4km/HadGEM2-ES_4km_anomaly_rcp85_1995-2300_v2.nc"],
                    "8km": [
                        "AIS/Atmospheric_forcing/HadGEM2-ES-RCP85-repeat/Regridded_8km/HadGEM2-ES_8km_anomaly_rcp85_1995-2300_v2.nc"]
                },
            },
            "NorESM1-M": {
                "RCP26-repeat": {
                    "4km": [
                        "AIS/Atmospheric_forcing/NorESM1-M_RCP26-repeat/Regridded_4km/NorESM1-M_4km_anomaly_rcp26_1995-2300_v3.nc"],
                    "8km": [
                        "AIS/Atmospheric_forcing/NorESM1-M_RCP26-repeat/Regridded_8km/NorESM1-M_8km_anomaly_rcp26_1995-2300_v2.nc"]
                },
                "RCP85-repeat": {
                    "4km": [
                        "AIS/Atmospheric_forcing/NorESM1-M_RCP85-repeat/Regridded_4km/NorESM1-M_4km_anomaly_1995-2300_v2.nc"],
                    "8km": [
                        "AIS/Atmospheric_forcing/NorESM1-M_RCP85-repeat/Regridded_8km/NorESM1-M_8km_anomaly_1995-2300_v2.nc"]
                },
            },
            "UKESM1-0-LL": {
                "SSP126": {
                    "4km": [
                        "AIS/Atmospheric_forcing/UKESM1-0-LL_ssp126/Regridded_4km/UKESM1-0-LL_4km_anomaly_ssp126_1995-2100.nc",
                        "AIS/Atmospheric_forcing/UKESM1-0-LL_ssp126/Regridded_4km/UKESM1-0-LL_4km_anomaly_ssp126_2101-2300.nc"],
                    "8km": [
                        "AIS/Atmospheric_forcing/UKESM1-0-LL_ssp126/Regridded_8km/UKESM1-0-LL_8km_anomaly_ssp126_1995-2100.nc",
                        "AIS/Atmospheric_forcing/UKESM1-0-LL_ssp126/Regridded_8km/UKESM1-0-LL_8km_anomaly_ssp126_2101-2300.nc"]
                },
                "SSP585": {
                    "4km": [
                        "AIS/Atmospheric_forcing/UKESM1-0-LL_ssp585/Regridding_4km/UKESM1-0-LL_4km_anomaly_ssp585_1995-2100.nc",
                        "AIS/Atmospheric_forcing/UKESM1-0-LL_ssp585/Regridding_4km/UKESM1-0-LL_4km_anomaly_ssp585_2101-2300.nc"],
                    "8km": [
                        "AIS/Atmospheric_forcing/UKESM1-0-LL_ssp585/Regridding_8km/UKESM1-0-LL_8km_anomaly_ssp585_1995-2100.nc",
                        "AIS/Atmospheric_forcing/UKESM1-0-LL_ssp585/Regridding_8km/UKESM1-0-LL_8km_anomaly_ssp585_2101-2300.nc"]
                },
                "SSP585-repeat": {
                    "4km": [
                        "AIS/Atmospheric_forcing/UKESM1-0-LL_ssp585-repeat/Regridding_4km/UKESM1-0-LL_4km_anomaly_ssp585_1995-2300_v2.nc"],
                    "8km": [
                        "AIS/Atmospheric_forcing/UKESM1-0-LL_ssp585-repeat/Regridding_8km/UKESM1-0-LL_8km_anomaly_ssp585_1995-2300_v2.nc"]
                }
            }
        }
    }
