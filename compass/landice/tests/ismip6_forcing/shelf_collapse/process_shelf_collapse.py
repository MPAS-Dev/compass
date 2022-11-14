import os
import shutil
import numpy as np
import xarray as xr
from compass.landice.tests.ismip6_forcing.atmosphere.create_mapfile_smb \
    import build_mapping_file
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from compass.step import Step


class ProcessShelfCollapse(Step):
    """
    A step for processing (combine, remap and rename) the ISMIP6 shelf-collapse
    mask data
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip6_forcing.shelf_collapse.
                    ShelfCollapse
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name="process_shelf_collapse",
                         ntasks=4, min_tasks=1)

    def setup(self):
        """
        Set up this step of the test case
        """
        config = self.config
        section = config["ismip6_ais"]
        base_path_ismip6 = section.get("base_path_ismip6")
        base_path_mali = section.get("base_path_mali")
        period_endyear = section.get("period_endyear")
        model = section.get("model")
        scenario = section.get("scenario")
        res_ismip6 = section.get("res_ismip6")
        mali_mesh_name = section.get("mali_mesh_name")
        mali_mesh_file = section.get("mali_mesh_file")

        model_list_2300 = ["CCSM4", "CESM2-WACCM","HadGEM2-ES","UKESM1-0-LL"]
        if period_endyear != 2300 and model not in model_list_2300:
            raise ValueError(f"ice shelf-collapse masks are provided by the "
                             f"ISMIP6 only for CCSM4-RCP85, "
                             f"CESM2-WACCM-SSP585,HadGEM2-ES-RCP85, "
                             f"UKESM1-0-LL-SSP585 model&scenarios. Please "
                             f"set up the correct config options in "
                             f"the config file.")

        input_file = self._files[period_endyear][model][scenario] \
                     [res_ismip6][0]
        self.add_input_file(filename=mali_mesh_file,
                            target=os.path.join(base_path_mali,
                                                mali_mesh_file))
        self.add_input_file(filename=os.path.basename(input_file),
                            target=os.path.join(base_path_ismip6,
                                                input_file))
        self.add_output_file(filename=f"{mali_mesh_name}_"
                                      f"{os.path.basename(input_file)}")

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        section = config["ismip6_ais"]
        period_endyear = section.get("period_endyear")
        output_base_path = section.get("output_base_path")
        model = section.get("model")
        scenario = section.get("scenario")
        res_ismip6 = section.get("res_ismip6")
        mali_mesh_name = section.get("mali_mesh_name")
        mali_mesh_file = section.get("mali_mesh_file")

        # we always want neareststod for the remapping method because we want
        # a value between 0 and 1 per mesh point.
        method_remap = "neareststod"

        # interpolate and rename the data
        # ismip6 input files
        input_file = self._files[period_endyear][model][scenario] \
                     [res_ismip6][0]

        logger.info(f"!---Start processing the file---!")
        logger.info(f"processing the input file "
                    f"'{os.path.basename(input_file)}'")

        # temporary file name.
        remapped_file_temp = f"remapped.nc"

        # remap the input forcing file.
        logger.info(f"Calling the remapping function...")
        self.remap_ismip6_shelf_mask_to_mali_vars(os.path.basename(input_file),
                                                  remapped_file_temp,
                                                  mali_mesh_name,
                                                  mali_mesh_file,
                                                  method_remap)

        output_file = f"{mali_mesh_name}_{os.path.basename(input_file)}"

        # rename the ismip6 variables to MALI variables
        logger.info(f"Renaming the ismip6 variables to "
                    f"mali variable names...")
        self.rename_ismip6_shelf_mask_to_mali_vars(remapped_file_temp,
                                                   output_file)

        # round up/down the mask values to 1/0
        args = ["ncap2", "-O", "-s",
                "where(calvingMask>=0.5) calvingMask=1",
                output_file]

        check_call(args, logger=self.logger)

        args = ["ncap2", "-O", "-s",
                "where(calvingMask<0.5) calvingMask=0", output_file]
        check_call(args, logger=self.logger)

        logger.info(f"Remapping and renaming process done successfully. "
                    f"Removing the temporary files...")

        # remove the temporary combined file
        os.remove(remapped_file_temp)

        # place the output file in appropriate directory
        output_path = f"{output_base_path}/shelf_collapse/" \
                      f"{model}_{scenario}/"

        if not os.path.exists(output_path):
            logger.info("Creating a new directory for the output data...")
            os.makedirs(output_path)

        src = os.path.join(os.getcwd(), output_file)
        dst = os.path.join(output_path, output_file)
        shutil.copy(src, dst)

        logger.info(f"!---Done processing the current file---!")
        logger.info(f"")
        logger.info(f"")

    def remap_ismip6_shelf_mask_to_mali_vars(self, input_file, output_file,
                                             mali_mesh_name, mali_mesh_file,
                                             method_remap):
        """
        Remap the input ismip6 ice shelf-collapse mask data onto mali mesh

        Parameters
        ----------
        input_file: str
            data file on the ismip6 grid

        output_file : str
            ismip6 data remapped onto mali mesh

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

    def rename_ismip6_shelf_mask_to_mali_vars(self, remapped_file_temp,
                                              output_file):
        """
        Rename variables in the remapped ismip6 input data
        to the ones that MALI uses.

        Parameters
        ----------
        remapped_file_temp : str
            temporary ismip6 data remapped on mali mesh where data values are
            rounded up/down

        output_file : str
            remapped ismip6 data renamed on mali mesh
        """

        # open dataset in 20 years chunk
        ds = xr.open_dataset(remapped_file_temp, chunks=dict(time=20))

        # build dictionary and rename the ismip6 dimension and variables
        ismip6_to_mali_dims = dict(
            ncol="nCells",
            time="Time")
        ds = ds.rename(ismip6_to_mali_dims)

        ismip6_to_mali_vars = dict(mask="calvingMask")
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

        ds["calvingMask"] = ds["calvingMask"].astype(int)

        # drop unnecessary variables on the regridded data on MALI mesh
        ds = ds.drop_vars(["lat_vertices", "lon_vertices","lat",
                           "lon", "area"])

        # write to a new netCDF file
        write_netcdf(ds, output_file)
        ds.close()

    # input files: input uniform melt rate coefficient (gamma0)
    _files = {
        #"2100":  Will be added later
        "2300": {
            "CCSM4": {
                "RCP85": {
                    "8km":[
                        "AIS/ShelfCollapse_forcing/CCSM4_RCP85/ice_shelf_collapse_mask_CCSM4_RCP85_1995-2300_08km.nc"],
                    "4km":[
                        "AIS/ShelfCollapse_forcing/CCSM4_RCP85/ice_shelf_collapse_mask_CCSM4_RCP85_1995-2300_04km.nc"]
                },
            },
            "CESM2-WACCM": {
                "SSP585": {
                    "8km":[
                        "AIS/ShelfCollapse_forcing/CESM2_WACCM_ssp585/ice_shelf_collapse_mask_CESM2_WACCM_ssp585_1995-2300_08km.nc"],
                    "4km":[
                        "AIS/ShelfCollapse_forcing/CESM2_WACCM_ssp585/ice_shelf_collapse_mask_CESM2_WACCM_ssp585_1995-2300_04km.nc"]
                },
            },
            "HadGEM2-ES": {
                "RCP85": {
                    "8km":[
                        "AIS/ShelfCollapse_forcing/HadGEM2-ES_RCP85/ice_shelf_collapse_mask_HadGEM2-ES_RCP85_1995-2300_08km.nc"],
                    "4km":[
                        "AIS/ShelfCollapse_forcing/HadGEM2-ES_RCP85/ice_shelf_collapse_mask_HadGEM2-ES_RCP85_1995-2300_04km.nc"]
                },
            },
            "UKESM1-0-LL": {
                "SSP585": {
                    "8km":[
                        "AIS/ShelfCollapse_forcing/UKESM1-0-LL_ssp585/ice_shelf_collapse_mask_UKESM1-0-LL_ssp585_1995-2300_08km.nc"],
                    "4km":[
                        "AIS/ShelfCollapse_forcing/UKESM1-0-LL_ssp585/ice_shelf_collapse_mask_UKESM1-0-LL_ssp585_1995-2300_04km.nc"]
                }
            }
        }
    }
