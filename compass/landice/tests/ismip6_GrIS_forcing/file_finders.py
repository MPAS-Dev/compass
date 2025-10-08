import glob
import os

import xarray as xr
from xarray.coders import CFDatetimeCoder

# create mapping dictionary of ISMIP6 variables to MALI variable names
{"thermal_forcing": "ismip6_2dThermalForcing",
 "basin_runoff": "basin_runoff"}


def check_year(fn, start=2015, end=2100):

    # strip the file extension from filename and
    # convert the year at end of str to an int
    fn_year = int(os.path.splitext(fn)[0].split("-")[-1])

    # check if year is within range
    return start <= fn_year <= end


class oceanFileFinder:

    def __init__(self, archive_fp, version="v4"):

        self.version = version

        # file strucutre within Ocean_Forcing directory for navigating
        file_struct = f"Ocean_Forcing/Melt_Implementation/{version}"

        # file path to directory containing forcings files from each GCM
        dir2GCMs = os.path.join(archive_fp, file_struct)

        # should do some checking here that all the fps work...
        self.dir2GCMs = dir2GCMs

    def get_filename(self, GCM, scenario, variable):

        # convert from variable name within NetCDF file, to variable name as it
        # appears in the filename
        if variable == "thermal_forcing":
            fn_var = "oceanThermalForcing"
        elif variable == "basin_runoff":
            fn_var = "basinRunoff"
        else:
            raise ValueError(f"invalid varibale name: {variable}")

        # within filename scenario have for removed
        scenario_no_dot = "".join(scenario.split("."))

        fp = (f"{GCM.lower()}_{scenario}/"
              f"MAR3.9_{GCM}_{scenario_no_dot}_{fn_var}_{self.version}.nc")

        # check that file exists!!!
        return os.path.join(self.dir2GCMs, fp)


class atmosphereFileFinder:

    def __init__(self, archive_fp, version="v1", workdir="./"):

        self.version = version

        if os.path.exists(workdir):
            self.workdir = workdir
        else:
            print("gotta do error checking")

        # file strucutre within Atmosphere_Forcing directory for navigating
        file_struct = f"Atmosphere_Forcing/aSMB_observed/{version}"

        # file path to directory containing forcings files from each GCM
        dir2GCMs = os.path.join(archive_fp, file_struct)

        # should do some checking here that all the fps work...
        self.dir2GCMs = dir2GCMs

    def get_filename(self, GCM, scenario, variable, start=1950, end=2100):
        """
        """
        if GCM == "NorESM1-M":
            GCM = "NorESM1"
        # get a list of all the yearly files within the period of intrest
        yearly_files = self.__find_yearly_files(GCM, scenario, variable,
                                                start, end)

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
            self.__combine_files(yearly_files, out_fp)

        return out_fp

    def __find_yearly_files(self, GCM, scenario, variable, start, end):
        """
        """
        # within filename scenario have for removed
        scenario_no_dot = "".join(scenario.split("."))

        # create filename template to be globbed
        fp = (f"{GCM}-{scenario_no_dot}/{variable}/"
              f"{variable}_MARv3.9-yearly-{GCM}-{scenario_no_dot}-*.nc")

        # glob the files
        all_files = glob.glob(os.path.join(self.dir2GCMs, fp))
        # all directories should have same number of files; so check here to
        # make sure things worked properly

        # get the files within the desired period
        files = sorted(f for f in all_files if check_year(f, start, end))

        # make sure the length of fns returned matches the length (in years)
        # of the period
        period = (end - start) + 1

        assert len(files) == period
        return files

    def __combine_files(self, files, out_fn):
        """
        """
        decoder = CFDatetimeCoder(use_cftime=True)

        ds = xr.open_mfdataset(files, decode_times=decoder,
                               concat_dim="time", combine="nested",
                               data_vars='minimal', coords='minimal',
                               compat="broadcast_equals",
                               combine_attrs="override", engine='netcdf4')

        ds.to_netcdf(out_fn, engine='netcdf4')
