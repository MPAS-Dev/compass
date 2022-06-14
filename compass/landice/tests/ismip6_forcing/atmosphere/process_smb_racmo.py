import os
import shutil
import subprocess
import xarray as xr
import warnings
from mpas_tools.io import write_netcdf
from compass.step import Step
from compass.landice.tests.ismip6_forcing.atmosphere.create_mapfile_smb \
    import build_mapping_file




class ProcessSmbRacmo(Step):
    """
    A step for processing the regional RACMO surface mass balance data
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
        super().__init__(test_case=test_case, name='process_smb_racmo',
                         cores=4, min_cores=1)

    def setup(self):
        """
        Set up this step of the test case
        """
        config = self.config
        section = config['ismip6_ais']
        base_path_mali = section.get('base_path_mali')
        mali_mesh_file = section.get('mali_mesh_file')

        section = config['ismip6_ais_atmosphere']
        process_smb_racmo = section.getboolean("process_smb_racmo")

        self.add_input_file(filename=mali_mesh_file,
                            target=os.path.join(base_path_mali,
                                                mali_mesh_file))

        if process_smb_racmo:
            base_path_racmo = section.get("base_path_racmo")
            input_file_list = self._files
            self.add_input_file(filename=input_file_list[0],
                                target=os.path.join(base_path_racmo,
                                                    input_file_list[0]))
            output_file = f"processed_RACMO2.3p2_ANT27" \
                          f"_smb_climatology_1995-2017.nc"

            self.add_output_file(filename=output_file)
        else:
            warnings.warn(f"'process_smb_racmo' is set to 'False'. This step"
                          f" will not run unless set 'True' in the"
                          f" config file.")


    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config

        section = config["ismip6_ais"]
        mali_mesh_name = section.get("mali_mesh_name")
        mali_mesh_file = section.get("mali_mesh_file")
        output_base_path = section.get("output_base_path")

        section = config["ismip6_ais_atmosphere"]
        method_remap = section.get("method_remap")
        process_modern_smb = section.getboolean("process_smb_racmo")

        input_file_list = self._files
        racmo_file_temp1 = "RACMO2.3p2_smb_climatology_1995_2017.nc"
        racmo_file_temp2 = "RACMO2.3p2_smb_climatology_1995_2017_" \
                           "correct_unit.nc"
        output_file = f"processed_RACMO2.3p2_ANT27" \
                      f"_smb_climatology_1995-2017.nc"

        output_file_final = os.path.join(output_base_path, output_file)

        if os.path.exists(output_file_final):
            print("Processed RACMO SMB data already exists in the output path"
                  f"{output_base_path}. Not processing the source data...")
            return

        input_file_list = self._files
        # take the time average over the period 1995-2017
        args = ["ncra", "-O", "-F", "-d", "time,17,39",
                f"{input_file_list[0]}",
                f"{racmo_file_temp1}"]

        subprocess.check_call(args)

        # interpolate the racmo smb data
        remapped_file_temp = "remapped.nc"  # temporary file name

        # call the function that reads in, remap and rename the file.
        logger.info("Calling a remapping function...")
        self.remap_source_smb_to_mali(f"{racmo_file_temp1}",
                                      remapped_file_temp,
                                      mali_mesh_name,
                                      mali_mesh_file,
                                      method_remap)

        # perform algebraic operation on the source data in unit of kg/m^2
        # to be in unit of kg/m^2/s
        args = ["ncap2", "-O", "-v", "-s",
                "sfcMassBal=smb/(60*60*24*365)",
                f"{remapped_file_temp}",
                f"{racmo_file_temp2}"]

        subprocess.check_call(args)

        # change the unit attribute to kg/m^2/s
        args = ["ncatted", "-O", "-a",
                "units,sfcMassBal,m,c,'kg m-2 s-1'",
                f"{racmo_file_temp2}"]

        subprocess.check_call(args)

        # call the function that renames the ismip6 variables to MALI variables
        logger.info("Renaming source variables to mali variable names...")

        self.rename_source_smb_to_mali_vars(racmo_file_temp2, output_file)

        logger.info("Processing done successfully. "
                    "Removing the temporary files...")
        # remove the temporary remapped and combined files
        os.remove(remapped_file_temp)
        os.remove(racmo_file_temp1)
        os.remove(racmo_file_temp2)

        # place the output file in appropriate directory
        output_path = f'{output_base_path}/atmosphere_forcing/' \
                      f'RACMO_climatology_1995-2017'
        if not os.path.exists(output_path):
            print("Creating a new directory for the output data")
            os.makedirs(output_path)

        src = os.path.join(os.getcwd(), output_file)
        dst = os.path.join(output_path, output_file)
        shutil.copy(src, dst)

    def remap_source_smb_to_mali(self, input_file, output_file, mali_mesh_name,
                                 mali_mesh_file, method_remap):
        """
        Remap the input ismip6 smb forcing data onto mali mesh

        Parameters
        ----------
        input_file: str
            input racmo smb data on its native rotaed pole grid
        output_file : str
            smb data remapped on mali mesh
        mali_mesh_name : str
            name of the mali mesh used to name mapping files
        mali_mesh_file : str, optional
            The MALI mesh file if mapping file does not exist
        method_remap : str, optional
            Remapping method used in building a mapping file

        """
        mapping_file = f"map_racmo_24km_to_" \
                       f"{mali_mesh_name}_{method_remap}.nc"

        # check if a mapfile exists
        if not os.path.exists(mapping_file):
            # build a mapping file if it doesn't already exist
            self.logger.info(f"Creating a mapping file. "
                             f"Mapping method used: {method_remap}")
            build_mapping_file(self.config, self.cores, self.logger,
                               input_file, mapping_file, mali_mesh_file,
                               method_remap)
        else:
            self.logger.info("Mapping file exists. "
                             "Remapping the input data...")

        args = ["ncremap",
                "-i", input_file,
                "-o", output_file,
                "-m", mapping_file,
                "-v", "smb"]

        subprocess.check_call(args)

    def rename_source_smb_to_mali_vars(self, remapped_file_temp, output_file):
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

        # add xtime variable
        ds["xtime"] = ("Time", ["1995-01-01_00:00:00".ljust(64)])
        ds["xtime"] = ds.xtime.astype('S')

        # drop unnecessary variables
        ds = ds.drop_vars(["height"])

        # squeeze unnecessary coordinate variable
        ds["sfcMassBal"] = ds["sfcMassBal"].squeeze(dim="height")

        # write to a new netCDF file
        write_netcdf(ds, output_file)
        ds.close()

    def correct_SMB_anomaly_for_baseSMB(self, output_file, mali_mesh_file):
        """
        Apply the MALI base SMB to the ismip6 SMB anomaly field

        Parameters
        ----------
        output_file : str
            remapped ismip6 data renamed on mali mesh
        mali_mesh_file : str
            initialized MALI mesh file in which the base SMB field exists
        """

        ds = xr.open_dataset(output_file)
        ds_base = xr.open_dataset(mali_mesh_file)
        # get the first time index
        ref_smb = ds_base["sfcMassBal"].isel(Time=0)
        # correct for the reference smb
        ds["sfcMassBal"] = ds["sfcMassBal"] + ref_smb

        # write to a new netCDF file
        write_netcdf(ds, output_file)
        ds.close()

    # create a dictionary for the regional climate RACMO dataset and the
    _files= ["RACMO2.3p2_ANT27_smb_yearly_1979_2018.nc"]
