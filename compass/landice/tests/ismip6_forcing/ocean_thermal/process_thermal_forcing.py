import os
import numpy as np
import shutil
import subprocess
import xarray as xr
from compass.landice.tests.ismip6_forcing.create_mapfile \
    import build_mapping_file
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from compass.step import Step


class ProcessThermalForcing(Step):
    """
    A step for creating a mesh and initial condition for dome test cases

    Attributes
    ----------
    process_obs : bool
        Whether we are processing observations rather than CMIP model data
    """
    def __init__(self, test_case, process_obs):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip6_forcing.ocean_thermal.
                    OceanThermal
            The test case this step belongs to

        process_obs : bool
            Whether we are processing observations rather than CMIP model data
        """
        super().__init__(test_case=test_case, name="process_thermal_forcing",
                         ntasks=4, min_tasks=1)
        self.process_obs = process_obs

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
        model = section.get("model")
        scenario = section.get("scenario")

        process_obs_data = self.process_obs

        self.add_input_file(filename=mali_mesh_file,
                            target=os.path.join(base_path_mali,
                                                mali_mesh_file))

        if process_obs_data:
            input_file = self._file_obs
            output_file = f"{mali_mesh_name}_obs_TF_1995-2017_8km_x_60m.nc"
        else:
            input_file = self._files[period_endyear][model][scenario]
            output_file = f"{mali_mesh_name}_TF_{model}_{scenario}_" \
                          f"{period_endyear}.nc"

        self.add_input_file(filename=os.path.basename(input_file[0]),
                            target=os.path.join(base_path_ismip6,
                            input_file[0]))
        self.add_output_file(filename=output_file)

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
        output_base_path = section.get("output_base_path")

        section = config["ismip6_ais_ocean_thermal"]
        method_remap = section.get("method_remap")
        process_obs_data = self.process_obs

        if process_obs_data:
            input_file_list = self._file_obs
            output_file = f"{mali_mesh_name}_obs_TF_1995-2017_8km_x_60m.nc"
            output_path = f"{output_base_path}/ocean_thermal_forcing/"\
                          f"obs"
        else:
            input_file_list = self._files[period_endyear][model][scenario]
            output_file = f"{mali_mesh_name}_TF_" \
                          f"{model}_{scenario}_{period_endyear}.nc"
            output_path = f"{output_base_path}/ocean_thermal_forcing/" \
                          f"{model}_{scenario}/1995-{period_endyear}"

        input_file = os.path.basename(input_file_list[0])

        # interpolate and rename the ismip6 thermal forcing data
        remapped_file_temp = "remapped.nc"  # temporary file name

        # call the function that reads in, remap and rename the file.
        logger.info("Calling the remapping function...")
        self.remap_ismip6_thermal_forcing_to_mali_vars(input_file,
                                                       remapped_file_temp,
                                                       mali_mesh_name,
                                                       mali_mesh_file,
                                                       method_remap)

        # call the function that renames the ismip6 variables to MALI variables
        logger.info(f"Renaming the ismip6 variables to mali variable names...")
        self.rename_ismip6_thermal_forcing_to_mali_vars(remapped_file_temp,
                                                        output_file)

        logger.info(f"Remapping and renamping process done successfully. "
                    f"Removing the temporary file 'remapped.nc'...")

        # remove the temporary combined file
        os.remove(remapped_file_temp)

        # place the output file in appropriate directory
        if not os.path.exists(output_path):
            logger.info(f"Creating a new directory for the output data...")
            os.makedirs(output_path)

        src = os.path.join(os.getcwd(), output_file)
        dst = os.path.join(output_path, output_file)
        shutil.copy(src, dst)

        logger.info(f"!---Done processing the file---!")

    def remap_ismip6_thermal_forcing_to_mali_vars(self,
                                                  input_file,
                                                  output_file,
                                                  mali_mesh_name,
                                                  mali_mesh_file,
                                                  method_remap):
        """
        Remap the input ismip6 thermal forcing data onto mali mesh

        Parameters
        ----------
        input_file: str
            ismip6 thermal forcing data on its native polarstereo 8km grid

        output_file : str
            ismip6 data remapped on mali mesh

        mali_mesh_name : str
            name of the mali mesh used to name mapping files

        mali_mesh_file : str, optional
            The MALI mesh file if mapping file does not exist

        method_remap : str, optional
            Remapping method used in building a mapping file
        """

        # check if a mapfile
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

    def rename_ismip6_thermal_forcing_to_mali_vars(self,
                                                   remapped_file_temp,
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

        config = self.config
        process_obs_data = self.process_obs

        # open dataset in 20 years chunk
        ds = xr.open_dataset(remapped_file_temp, chunks=dict(time=20),
                             engine="netcdf4")

        ds["ismip6shelfMelt_zOcean"] = ds.z
        ds = ds.drop_vars("z")  # dropping 'z' while it's still called 'z'

        # build dictionary for ismip6 variables that MALI takes in
        if process_obs_data:
            ismip6_to_mali_dims = dict(
                z="nISMIP6OceanLayers",
                ncol="nCells")
            ds["thermal_forcing"] = ds["thermal_forcing"].expand_dims(
                dim="Time", axis=0)
            ds = ds.rename(ismip6_to_mali_dims)
        else:
            ismip6_to_mali_dims = dict(
                z="nISMIP6OceanLayers",
                time="Time",
                ncol="nCells")
            ds = ds.rename(ismip6_to_mali_dims)
            # add xtime variable
            xtime = []
            for t_index in range(ds.sizes["Time"]):
                date = ds.Time[t_index]
                # forcing files do not all match even years,
                # so round up the years
                # pandas round function does not work for years,
                # so do it manually
                yr = date.dt.year.values
                mo = date.dt.month.values
                dy = date.dt.day.values
                dec_yr = np.around(yr + (30 * (mo - 1) + dy) / 365.0)
                date = f"{dec_yr.astype(int)}-01-01_00:00:00".ljust(64)
                xtime.append(date)

            ds["xtime"] = ("Time", xtime)
            ds["xtime"] = ds.xtime.astype('S')
            ds = ds.drop_vars(["Time"])

        ismip6_to_mali_vars = dict(
            thermal_forcing="ismip6shelfMelt_3dThermalForcing")
        ds = ds.rename(ismip6_to_mali_vars)

        # drop unnecessary variables
        ds = ds.drop_vars(["z_bnds", "lat_vertices", "area",
                           "lon_vertices", "lat", "lon"])

        # transpose dimension
        ds["ismip6shelfMelt_3dThermalForcing"] = \
            ds["ismip6shelfMelt_3dThermalForcing"].transpose(
            "Time", "nCells", "nISMIP6OceanLayers")

        # write to a new netCDF file
        write_netcdf(ds, output_file)
        ds.close()

    # create a nested dictionary for the ISMIP6 original forcing files including relative path
    _file_obs = ["AIS/Ocean_Forcing/climatology_from_obs_1995-2017/obs_thermal_forcing_1995-2017_8km_x_60m.nc"]
    _files = {
        "2100": {
            "CCSM4": {
                "RCP85": ["AIS/Ocean_Forcing/ccsm4_rcp8.5/1995-2100/CCSM4_thermal_forcing_8km_x_60m.nc"]
            },
            "CESM2": {
                "SSP585v1": [
                    "AIS/Ocean_Forcing/cesm2_ssp585/1995-2100/CESM2_ssp585_thermal_forcing_8km_x_60m.nc"],
                "SSP585v2": [
                    "AIS/Ocean_Forcing/cesm2_ssp585/1995-2100/CESM2_ssp585_thermal_forcing_8km_x_60m.nc"],
            },
            "CNRM_CM6": {
                "SSP126": [
                    "AIS/Ocean_Forcing/cnrm-cm6-1_ssp126/1995-2100/CNRM-CM6-1_ssp126_thermal_forcing_8km_x_60m.nc"],
                "SSP585": [
                    "AIS/Ocean_Forcing/cnrm-cm6-1_ssp585/1995-2100/CNRM-CM6-1_ssp585_thermal_forcing_8km_x_60m.nc"]
            },
            "CNRM_ESM2": {
                "SSP585": [
                    "AIS/Ocean_Forcing/cnrm-esm2-1_ssp585/1995-2100/CNRM-ESM2-1_ssp585_thermal_forcing_8km_x_60m.nc"]
            },
            "CSIRO-Mk3-6-0": {
                "RCP85": [
                    "AIS/Ocean_Forcing/csiro-mk3-6-0_rcp8.5/1995-2100/CSIRO-Mk3-6-0_RCP85_thermal_forcing_8km_x_60m.nc"]
            },
            "HadGEM2-ES": {
                "RCP85": [
                    "AIS/Ocean_Forcing/hadgem2-es_rcp8.5/1995-2100/HadGEM2-ES_RCP85_thermal_forcing_8km_x_60m.nc"]
            },
            "IPSL-CM5A-MR": {
                "RCP26": [
                    "AIS/Ocean_Forcing/ipsl-cm5a-mr_rcp2.6/1995-2100/IPSL-CM5A-MR_RCP26_thermal_forcing_8km_x_60m.nc"],
                "RCP85": [
                    "AIS/Ocean_Forcing/ipsl-cm5a-mr_rcp8.5/1995-2100/IPSL-CM5A-MR_RCP85_thermal_forcing_8km_x_60m.nc"]
            },
            "MIROC-ESM-CHEM": {
                "RCP85": [
                    "AIS/Ocean_Forcing/miroc-esm-chem_rcp8.5/1995-2100/MIROC-ESM-CHEM_thermal_forcing_8km_x_60m.nc"]
            },
            "NorESM1-M": {
                "RCP26": [
                    "AIS/Ocean_Forcing/noresm1-m_rcp2.6/1995-2100/NorESM1-M_RCP26_thermal_forcing_8km_x_60m.nc"],
                "RCP85": [
                    "AIS/Ocean_Forcing/noresm1-m_rcp8.5/1995-2100/NorESM1-M_thermal_forcing_8km_x_60m.nc"]
            },
            "UKESM1-0-LL": {
                "SSP585": [
                    "AIS/Ocean_Forcing/ukesm1-0-ll_ssp585/1995-2100/UKESM1-0-LL_ssp585_thermal_forcing_8km_x_60m.nc"]
            }
        },
        "2300": {
            "CCSM4": {
                "RCP85": [
                    "AIS/Ocean_forcing/ccsm4_RCP85/1995-2300/CCSM4_RCP85_thermal_forcing_8km_x_60m.nc"]
            },
            "CESM2-WACCM": {
                "SSP585": [
                    "AIS/Ocean_forcing/cesm2-waccm_ssp585/1995-2299/CESM2-WACCM_SSP585_thermal_forcing_8km_x_60m.nc"],
                "SSP585-repeat": [
                    "AIS/Ocean_forcing/cesm2-waccm_ssp585-repeat/1995-2300/CESM2-WACCM_ssp585_thermal_forcing_8km_x_60m.nc"]
            },
            "HadGEM2-ES": {
                "RCP85": [
                    "AIS/Ocean_forcing/hadgem2-es_RCP85/1995-2299/HadGEM2-ES_RCP85_thermal_forcing_8km_x_60m.nc"],
                "RCP85-repeat": [
                    "AIS/Ocean_forcing/hadgem2-es_RCP85-repeat/1995-2300/HadGEM2-ES_rcp85_thermal_forcing_8km_x_60m.nc"]
            },
            "NorESM1-M": {
                "RCP26-repeat": [
                    "AIS/Ocean_forcing/noresm1-m_RCP26-repeat/1995-2300/NorESM1-M_RCP26_thermal_forcing_8km_x_60m.nc"],
                "RCP85-repeat": [
                    "AIS/Ocean_forcing/noresm1-m_RCP85-repeat/1995-2300/NorESM1-M_thermal_forcing_8km_x_60m.nc"]
            },
            "UKESM1-0-LL": {
                "SSP126": [
                    "AIS/Ocean_forcing/ukesm1-0-ll_ssp126/1995-2300/UKESM1-0-LL_thermal_forcing_8km_x_60m_v2.nc"],
                "SSP585": [
                    "AIS/Ocean_forcing/ukesm1-0-ll_ssp585/1995-2300/UKESM1-0-LL_SSP585_thermal_forcing_8km_x_60m.nc"],
                "SSP585-repeat": [
                    "AIS/Ocean_forcing/ukesm1-0-ll_ssp585-repeat/1995-2300/UKESM1-0-LL_ssp585_thermal_forcing_8km_x_60m.nc"]
            }
        }
    }
