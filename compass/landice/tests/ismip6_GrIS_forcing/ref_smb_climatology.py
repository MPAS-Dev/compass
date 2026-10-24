import os

import xarray as xr
from mpas_tools.io import write_netcdf

from compass.landice.tests.ismip6_GrIS_forcing.utilities import remap_variables
from compass.step import Step


class SMBRefClimatology(Step):
    """
    A step to produce a surface mass balance climatology from RACMO data

    Attributes
    ----------
    racmo_smb_fp: str
        File path to RACMO SMB data proccessed by this step
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        name = "smb_ref_climatology"
        subdir = None

        super().__init__(test_case=test_case, name=name, subdir=subdir)

        # attrs will set by the `setup` method, so user specified
        # config options can be parsed
        self.racmo_smb_fp = None

    def setup(self):
        """
        Parse config file for path to RACMO data

        Then add path to the climatology produced by this step as an attribute
        to the test case. Allowing climatology to be reused by other steps
        """

        # parse user specified parameters from the config
        config = self.config
        smb_ref_section = config["smb_ref_climatology"]

        racmo_directory = smb_ref_section.get("racmo_directory")
        # this filename should probably just be hardcoded.....
        racmo_smb_fn = smb_ref_section.get("racmo_smb_fn")
        # combine the directory and filename
        racmo_smb_fp = os.path.join(racmo_directory, racmo_smb_fn)

        # make sure the combined filename exists
        if not os.path.exists(racmo_smb_fp):
            # check if the parent directory exists
            if not os.path.exists(racmo_directory):
                raise FileNotFoundError(f"{racmo_directory} does not exist")
            # the parent directory exists but the forcing file does not
            else:
                raise FileNotFoundError(f"{racmo_smb_fp} does not exist")

        # add the racmo smb as an attribute and as an input file
        self.racmo_smb_fp = racmo_smb_fp
        self.add_input_file(filename=racmo_smb_fp)

        # get the start and end dates for the climatological mean
        clima_start = smb_ref_section.getint("climatology_start")
        clima_end = smb_ref_section.getint("climatology_end")

        # make a descriptive filename based on climatology period
        clima_fn = f"racmo_climatology_{clima_start}--{clima_end}.nc"
        # add the steps workdir to the filename to make it a full path
        clima_fp = os.path.join(self.work_dir, clima_fn)

        # set the testcase attribute and add it as an output file for this
        # step so the climatology will by useable by other steps
        self.test_case.smb_ref_climatology = clima_fp
        self.add_output_file(filename=self.test_case.smb_ref_climatology)

    def run(self):
        """
        Run this step of the test case
        """
        racmo_smb = self.racmo_smb_fp
        smb_ref_climatology = self.test_case.smb_ref_climatology
        racmo_2_mali_weights = self.test_case.racmo_2_mali_weights

        # parse user specified parameters from the config
        config = self.config
        smb_ref_section = config["smb_ref_climatology"]
        # get the start and end dates for the climatological mean
        clima_start = smb_ref_section.getint("climatology_start")
        clima_end = smb_ref_section.getint("climatology_end")

        # remap the gridded racmo data onto the mpas grid
        remap_variables(
            racmo_smb, smb_ref_climatology, racmo_2_mali_weights, ["SMB_rec"]
        )

        ds = xr.open_dataset(smb_ref_climatology, decode_times=False)

        # find indices of climatology start/end (TO DO: make more robust)
        s_idx = ((clima_start - 1958) * 12) - 1
        e_idx = ((clima_end - 1958) * 12) - 1

        # calculate climatology
        ds["SMB_rec"] = ds.SMB_rec.isel(time=slice(s_idx, e_idx)).mean("time")
        # rename variables to match MALI/MPAS conventiosn
        ds = ds.rename(SMB_rec="sfcMassBal", ncol="nCells")
        # drop unused dimensions
        ds = ds.drop_dims(["time", "nv"])
        # drop un-needed varibales
        ds = ds.drop_vars(["area", "lat", "lon"])
        # convert `sfcMassBal` to MPAS units
        ds["sfcMassBal"] /= (60 * 60 * 24 * 365) / 12.
        # add a units attribute to `sfcMassBal`
        ds["sfcMassBal"].attrs["units"] = "kg m-2 s-1"
        # expand sfcMassBal dimension to match what MALI expects
        ds["sfcMassBal"] = ds.sfcMassBal.expand_dims("Time")
        # write the file
        write_netcdf(ds, smb_ref_climatology)
