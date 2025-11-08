import glob
import os

import xarray as xr
from xarray.coders import CFDatetimeCoder


def check_year(fn, start=2015, end=2100):
    """
    Check is file is within desired period based on it's filename

    Parameters
    ----------
    fn: str
        Filename to check.

    start: int
        Start of period

    end: str
        End of period
    """

    # strip the file extension from filename and
    # convert the year at end of str to an int
    fn_year = int(os.path.splitext(fn)[0].split("-")[-1])

    # check if year is within range
    return start <= fn_year <= end


class ISMIP6FileFinder:
    """
    Base class for itterating over ISMIP6 forcing file archives

    Attributes
    ----------
    version: str
        Version of the ISMIP6 ocean archive filename are from

    dir_w_GCMs: str
        File path to directory containing the GCM data
    """
    def __init__(self, version, dir_w_GCMs):
        """
        Parameters
        ----------
        version: str
            Version of the ISMIP6 archive filename are from

        dir_w_GCMs: str
            File path to directory containing the GCM data
        """

        self.version = version

        self.dir_w_GCMs = self.check_file_exists(dir_w_GCMs)

    def get_filename(self):
        """
        Return filepath to variable for requested GCM, scenario, and period
        """
        raise NotImplementedError()

    def check_file_exists(self, fp):
        """
        Ensure the filepath constructed actually exists
        """

        if os.path.exists(fp):
            return fp
        else:
            msg = f"Cannot Find File: \n {fp} \n"
            raise FileNotFoundError(msg)


class oceanFileFinder(ISMIP6FileFinder):
    """
    Subclass for itterating ISMIP6 ocean archive
    """

    def __init__(self, archive_fp, version="v4"):
        """
        Parameters
        ----------
        version: str
            Version of the ISMIP6 archive filename are from

        dir_w_GCMs: str
            File path to directory containing the GCM data
        """

        # file strucutre within Ocean_Forcing directory for navigating
        file_struct = f"Ocean_Forcing/Melt_Implementation/{version}"

        # file path to directory containing forcings files from each GCM
        dir_w_GCMs = os.path.join(archive_fp, file_struct)

        super().__init__(version, dir_w_GCMs)

    def get_filename(self, GCM, scenario, variable):
        """
        Return filepath to variable for requested GCM and scenario

        Parameters
        ----------
        GCM: str
            General Circulation Model the forcing is derived from

        scenario: str
            Emissions scenario

        var: str
            Name of the variable to find file(s) for
        """

        # convert var name in NetCDF file, to var name in the filename
        if variable == "thermal_forcing":
            fn_var = "oceanThermalForcing"
        elif variable == "basin_runoff":
            fn_var = "basinRunoff"
        else:
            msg = f"invalid varibale name: {variable}"
            raise ValueError(msg)

        # within filename scenario have for removed
        scenario_no_dot = "".join(scenario.split("."))

        fp = (
            f"{GCM.lower()}_{scenario}/"
            f"MAR3.9_{GCM}_{scenario_no_dot}_{fn_var}_{self.version}.nc"
        )

        out_fp = os.path.join(self.dir2GCMs, fp)

        return self.check_file_exists(out_fp)


class atmosphereFileFinder:
    """
    Subclass for itterating ISMIP6 atmosphere archive
    """

    def __init__(self, archive_fp, version="v1", workdir="./"):
        """
        Parameters
        ----------
        version: str
            Version of the ISMIP6 archive filename are from

        dir_w_GCMs: str
            File path to directory containing the GCM data
        """

        # file strucutre within Atmosphere_Forcing directory for navigating
        file_struct = f"Atmosphere_Forcing/aSMB_observed/{version}"

        # file path to directory containing forcings files from each GCM
        dir_w_GCMs = os.path.join(archive_fp, file_struct)

        super().__init__(version, dir_w_GCMs)

        if os.path.exists(workdir):
            self.workdir = workdir
        else:
            print("gotta do error checking")

    def get_filename(self, GCM, scenario, variable, start=1950, end=2100):
        """
        Return filepath to variable for requested GCM, scenario and period

        Parameters
        ----------
        GCM: str
            General Circulation Model the forcing is derived from

        scenario: str
            Emissions scenario

        var: str
            Name of the variable to find file(s) for

        start: int
            First year to process forcing for

        end: int
            Final year to process forcing for
        """
        if GCM == "NorESM1-M":
            GCM = "NorESM1"

        # get a list of all the yearly files within the period of intrest
        yearly_files = self._find_yearly_files(
            GCM, scenario, variable, start, end
        )

        # still need to make the output filename to write combined files to
        out_fn = f"MAR3.9_{GCM}_{scenario}_{variable}_{start}--{end}.nc"
        # relative to the workdir, which we've already checked if if existed
        # make the filepath for the nested GCM/scenario/var direcotry struct.
        top_fp = os.path.join(self.workdir, f"{GCM}-{scenario}/{variable}")

        # make the nested directory strucure; if needed
        if not os.path.exists(top_fp):
            os.makedirs(top_fp, exist_ok=True)

        out_fp = os.path.join(top_fp, out_fn)

        # only combine the files if they don't exist yet
        if not os.path.exists(out_fp):
            # with output filename, combine the files to a single
            self._combine_files(yearly_files, out_fp)

        return out_fp

    def _find_yearly_files(self, GCM, scenario, variable, start, end):
        """
        Each year of atmospheric forcing data is written to seperate file.
        So, get a sorted list of files for the requested years
        """
        # within filename scenario have for removed
        scenario_no_dot = "".join(scenario.split("."))

        # create filename template to be globbed
        fp = (
            f"{GCM}-{scenario_no_dot}/{variable}/"
            f"{variable}_MARv3.9-yearly-{GCM}-{scenario_no_dot}-*.nc"
        )

        # glob the files
        all_files = glob.glob(os.path.join(self.dir2GCMs, fp))
        # all directories should have same number of files; so check here to
        # make sure things worked properly

        # get the files within the desired period
        files = sorted(f for f in all_files if check_year(f, start, end))
        # make sure that each individual file exists
        files = sorted(self.check_file_exists(f) for f in files)

        # number of fns returned  must match length of the period (in years)
        period = (end - start) + 1

        if len(files) != period:
            msg = (
                f"Number of yearly files found for: \n"
                f"\t GCM: {GCM}, Scenario: {scenario}, Period: {start}-{end}\n"
                f"for variable: {variable} does not match the lenth of the"
                f"period requested. Please investigate in: "
                f"\t {self.dir_w_GCMs}"
            )
            raise ValueError(msg)

        return files

    def _combine_files(self, files, out_fn):
        """
        Concatenate multiple files along their time dimension
        """
        decoder = CFDatetimeCoder(use_cftime=True)

        ds = xr.open_mfdataset(files, decode_times=decoder,
                               concat_dim="time", combine="nested",
                               data_vars='minimal', coords='minimal',
                               compat="broadcast_equals",
                               combine_attrs="override", engine='netcdf4')

        ds.to_netcdf(out_fn, engine='netcdf4')
