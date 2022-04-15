import os
import pandas as pd
import shutil
import subprocess
import xarray as xr
import numpy as np
from compass.landice.tests.ismip6_forcing.atmosphere.create_mapfile_smb import build_mapping_file
from mpas_tools.scrip.from_mpas import scrip_from_mpas
from mpas_tools.io import write_netcdf
from compass.step import Step


class ProcessSMB(Step):
    """
    A step for processing the ISMIP6 surface mass balance data
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip6_forcing.atmosphere.Atmosphere
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='process_smb')

    def setup(self):
        """
        Set up this step of the test case
        """
        config = self.config
        section = config['ismip6_ais_atmosphere']

        input_path = section.get('input_path')
        input_file = section.get('input_file')
        self.add_input_file(filename=input_file,
                            target=os.path.join(input_path, input_file))
        mali_mesh_file = section.get('mali_mesh_file')
        self.add_input_file(filename=mali_mesh_file,
                            target=os.path.join(input_path, mali_mesh_file))
        self.add_output_file(filename= f"output_{input_file}")

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        section = config['ismip6_ais_atmosphere']

        input_file = section.get('input_file')
        mali_mesh_name = section.get('mali_mesh_name')
        mali_mesh_file = section.get('mali_mesh_file')
        method_remap = section.get('method_remap')
        output_file = f"output_{input_file}"

        # interpolate and rename the ismip6 thermal forcing data
        remapped_file_temp = "remapped.nc"  # temporary file name

        # call the function that reads in, remap and rename the file.
        logger.info("Calling a remapping function...")
        self.remap_ismip6smb_to_mali(input_file, remapped_file_temp,
                                     mali_mesh_name, mali_mesh_file,
                                     method_remap)

        # call the function that renames the ismip6 variables to MALI variables
        logger.info("Renaming the ismip6 variables to mali variable names...")
        self.rename_ismip6smb_to_mali_vars(remapped_file_temp, output_file)

        logger.info("Remapping and renamping process done successfully. "
                    "Removing the temporary file 'remapped.nc'")

        # remove the temporary combined file
        os.remove(remapped_file_temp)

    def remap_ismip6smb_to_mali(self, input_file, output_file, mali_mesh_name,
                                mali_mesh_file, method_remap):
        """
        Remap the input ismip6 thermal forcing data onto mali mesh

        Parameters
        ----------
        input_file: str
            ismip6 smb data on its native polarstereo 8km grid
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
            self.logger.info("Mapping file exists. "
                             "Remapping the input data...")

        # remap the input data
        args = ["ncremap",
                "-i", input_file,
                "-o", output_file,
                "-m", mapping_file,
                "-v", "smb_anomaly"]

        subprocess.check_call(args)

    def rename_ismip6smb_to_mali_vars(self, remapped_file_temp, output_file):
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
            date = ds.Time[t_index].values
            date = pd.to_datetime(str(date))
            date = date.strftime("%Y-%m-%d_00:00:00").ljust(64)
            xtime.append(date)

        ds["xtime"] = ("Time", xtime)
        ds["xtime"] = ds.xtime.astype('S')

        # drop unnecessary variables
        ds = ds.drop_vars(["lon", "lon_vertices", "lat", "lat_vertices",
                           "area"])

        # write to a new netCDF file
        write_netcdf(ds, output_file)
        ds.close()
