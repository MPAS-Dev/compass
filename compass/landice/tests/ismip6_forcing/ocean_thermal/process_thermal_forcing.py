import os
import subprocess
import xarray as xr
from compass.landice.tests.ismip6_forcing.ocean_thermal.create_mapfile \
    import build_mapping_file
from mpas_tools.io import write_netcdf
from compass.step import Step


class ProcessThermalForcing(Step):
    """
    A step for creating a mesh and initial condition for dome test cases
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip6_forcing.ocean_thermal.
                    OceanThermal
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='process_thermal_forcing')

    def setup(self):
        """
        Set up this step of the test case
        """
        config = self.config
        section = config['ismip6_ais']
        base_path_ismip6 = section.get('base_path_ismip6')
        base_path_mali = section.get('base_path_mali')
        mali_mesh_file = section.get('mali_mesh_file')
        period_endyear = section.get("period_endyear")
        model = section.get("model")
        scenario = section.get("scenario")

        if period_endyear == "NotAvailable":
            raise ValueError("You need to supply a user config file, which "
                             "should contain the ismip6_ais "
                             "section with the period_endyear option")
        if model == "NotAvailable":
            raise ValueError("You need to supply a user config file, which "
                             "should contain the ismip6_ais "
                             "section with the model option")
        if scenario == "NotAvailable":
            raise ValueError("You need to supply a user config file, which "
                             "should contain the ismip6_ais "
                             "section with the scenario option")

        self.add_input_file(filename=mali_mesh_file,
                            target=os.path.join(base_path_mali,
                                                mali_mesh_file))


        input_file_list = self._files[period_endyear][model][scenario]
        for file in input_file_list:
            self.add_input_file(filename=os.path.basename(file),
                                target=os.path.join(base_path_ismip6, file))

        output_file = f"processed_TF_{model}_{scenario}_{period_endyear}.nc"
        self.add_output_file(filename=output_file)

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config

        section = config['ismip6_ais']
        mali_mesh_name = section.get('mali_mesh_name')
        mali_mesh_file = section.get('mali_mesh_file')
        period_endyear = section.get("period_endyear")
        model = section.get("model")
        scenario = section.get("scenario")

        section = config['ismip6_ais_ocean_thermal']
        method_remap = section.get('method_remap')

        input_file_list = self._files[period_endyear][model][scenario]
        input_file = os.path.basename(input_file_list[0])

        # interpolate and rename the ismip6 thermal forcing data
        remapped_file_temp = "remapped.nc"  # temporary file name

        # call the function that reads in, remap and rename the file.
        print("Calling a remapping function...")
        self.remap_ismip6thermalforcing_to_mali(input_file,
                                                remapped_file_temp,
                                                mali_mesh_name,
                                                mali_mesh_file, method_remap)

        output_file = f"processed_TF_{model}_{scenario}_{period_endyear}.nc"
        # call the function that renames the ismip6 variables to MALI variables
        print("Renaming the ismip6 variables to mali variable names...")
        self.rename_ismip6thermalforcing_to_mali_vars(remapped_file_temp,
                                                      output_file)

        print("Remapping and renamping process done successfully. "
              "Removing the temporary file 'remapped.nc'")

        # remove the temporary combined file
        os.remove(remapped_file_temp)

    def remap_ismip6thermalforcing_to_mali(self, input_file,
                                           output_file, mali_mesh_name,
                                           mali_mesh_file, method_remap):
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
            build_mapping_file(input_file, mapping_file, mali_mesh_file,
                               method_remap)
        else:
            print("Mapping file exists. Remapping the input data...")

        # remap the input data
        args = ["ncremap",
                "-i", input_file,
                "-o", output_file,
                "-m", mapping_file]

        subprocess.check_call(args)

    def rename_ismip6thermalforcing_to_mali_vars(self, remapped_file_temp,
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
        ds = xr.open_dataset(remapped_file_temp, chunks=dict(time=20),
                             engine="netcdf4")

        ds["ismip6shelfMelt_zOcean"] = ds.z
        ds = ds.drop_vars('z')  # dropping 'z' while it's still called 'z'

        # build dictionary for ismip6 variables that MALI takes in
        ismip6_to_mali_dims = dict(
            z="nISMIP6OceanLayers",
            time="Time",
            ncol="nCells")
        ds = ds.rename(ismip6_to_mali_dims)

        ismip6_to_mali_vars = dict(
            thermal_forcing="ismip6shelfMelt_3dThermalForcing")
        ds = ds.rename(ismip6_to_mali_vars)

        # add xtime variable
        xtime = []
        for t_index in range(ds.sizes["Time"]):
            date = ds.Time[t_index]
            date = date.dt.strftime("%Y-%m-%d_00:00:00")
            date = str(date.values).ljust(64)
            xtime.append(date)

        ds["xtime"] = ("Time", xtime)
        ds["xtime"] = ds.xtime.astype('S')

        # drop unnecessary variables
        ds = ds.drop_vars(["z_bnds", "lat_vertices", "Time",
                           "lon_vertices", "lat", "lon", "area"])

        # write to a new netCDF file
        write_netcdf(ds, output_file)
        ds.close()


    # create a nested dictionary for the ISMIP6 original forcing files including relative path
    _files = {
        "2100": {
            "CCSM4": {
               # "RCP26": ["AIS/Atmosphere_forcing/ccsm4_rcp2.6/Regridded_8km/CCSM4_8km_anomaly_rcp26_1995-2100.nc"],
                "RCP85": ["AIS/Ocean_Forcing/ccsm4_rcp8.5/1995-2100/CCSM4_thermal_forcing_8km_x_60m.nc"]
            },
            "CESM2": {
                "SSP585": [
                    "AIS/Ocean_Forcing/cesm2_ssp585/1995-2100/CESM2_ssp585_thermal_forcing_8km_x_60m.nc"]
              #  "SSP585v1": [
              #     "AIS/Atmosphere_Forcing/CESM2_ssp585/Regridded_8km/CESM2_anomaly_ssp585_1995-2100_8km_v1.nc"],
              #  "SSP585v2": [
              #      "AIS/Atmosphere_forcing/CESM2_ssp585/Regridded_8km/CESM2_anomaly_ssp585_1995-2100_8km_v2.nc"]
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
              #  "RCP26": [
              #      "AIS/Atmosphere_Forcing/miroc-esm-chem_rcp2.6/Regridded_8km/MIROC-ESM-CHEM_8km_anomaly_rcp26_1995-2100.nc"],
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
                "SSP585":[
                    "AIS/Ocean_Forcing/ukesm1-0-ll_ssp585/1995-2100/UKESM1-0-LL_ssp585_thermal_forcing_8km_x_60m.nc"]
            }
        },
        "2300": {
            "CCSM4": {
                "RCP85": [
                    "AIS/Ocean_forcing/ccsm4_RCP85/1995-2300/CCSM4_RCP85_thermal_forcing_8km_x_60m.nc"]
            },
            "CESM2-WACCM": {
                "SSP585":[
                    "AIS/Ocean_forcing/cesm2-waccm_ssp585/1995-2299/CESM2-WACCM_SSP585_thermal_forcing_8km_x_60m.nc"],
                "SSP585-repeat": [
                    "AIS/Ocean_forcing/cesm2-waccm_ssp585-repeat/1995-2300/CESM2-WACCM_ssp585_thermal_forcing_8km_x_60m.nc"]
            },
  #          "CSIRO-Mk3-6-0": {
  #              "RCP85": [
  #                  "AIS/Atmospheric_forcing/CSIRO-Mk3-6-0_RCP85/Regridded_8km/CSIRO-Mk3-6-0_8km_anomaly_rcp85_1995-2100.nc",
  #                  "AIS/Atmospheric_forcing/CSIRO-Mk3-6-0_RCP85/Regridded_8km/CSIRO-Mk3-6-0_8km_anomaly_rcp85_2101-2300.nc"]
#         },
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
                    "AIS/Ocean_forcing/ukesm1-0-ll_ssp126/1995-2300/UKESM1-0-LL_thermal_forcing_8km_x_60m.nc"],
                "SSP585": [
                    "AIS/Ocean_forcing/ukesm1-0-ll_ssp585/1995-2300/UKESM1-0-LL_SSP585_thermal_forcing_8km_x_60m.nc"],
                "SSP585-repeat": [
                    "AIS/Ocean_forcing/ukesm1-0-ll_ssp585-repeat/1995-2300/UKESM1-0-LL_ssp585_thermal_forcing_8km_x_60m.nc"]
            }
        }
    }
