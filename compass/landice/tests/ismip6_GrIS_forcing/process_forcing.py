import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call

from compass.step import Step


def datetime_2_xtime(ds, var="Time", xtime_fmt="%Y-%m-%d_%H:%M:%S"):

    return xr.apply_ufunc(lambda t: t.strftime(xtime_fmt).ljust(64),
                          ds[var], vectorize=True, output_dtypes=["S"])


renaming_dict = {"thermal_forcing": "ismip6_2dThermalForcing",
                 "basin_runoff": "ismip6Runoff",
                 "aSMB": "sfcMassBal",
                 "dSMBdz": "sfcMassBal_lapseRate",
                 "aST": "surfaceAirTemperature",
                 "dSTdz": "surfaceAirTemperature_lapseRate"}


class ProcessForcing(Step):
    """

    """

    def __init__(self, test_case):
        """
        """

        name = "process_forcing"

        super().__init__(test_case=test_case, name=name, subdir=None,
                         cpus_per_task=1, min_cpus_per_task=1)
        # read and store the experiment dictionary

        # initalize the FileFinders are store them as dict?

    def run(self):

        # list of variables to process
        atmosphere_vars = ["aSMB", "aST", "dSMBdz", "dSTdz"]
        ocean_vars = ["basin_runoff", "thermal_forcing"]

        # loop over experiments passed via config file
        for expr_name, expr_dict in self.test_case.experiments.items():
            # print to logger which expriment we are on (i/n)

            GCM = expr_dict["GCM"]
            scenario = expr_dict["Scenario"]
            start = expr_dict["start"]
            end = expr_dict["end"]

            # need a special way to treat the RACMO datasets
            if expr_name in ["ctrl", "hist"]:
                continue

            ocean_forcing_datastes = []
            # loop over ocean variables seperately
            for var in ocean_vars:
                var_fp = self.test_case.findForcingFiles(GCM, scenario, var)

                ocean_forcing_datastes.append(
                    self.process_variable(GCM, scenario, var,
                                          var_fp, start, end))
                ocean_forcing_ds = xr.combined_by_coords(
                    ocean_forcing_datastes, combine_attrs="drop_conflicts")

            write_netcdf(ocean_forcing_ds, "ocean_forcing_test.nc")

            # loop over atmosphere variables
            for var in atmosphere_vars:
                # this pretty hacky with the tuple, fix the file finder
                var_fp, _ = self.test_case.findForcingFiles(GCM, scenario, var)

                self.process_variable(GCM, scenario, var, var_fp, start, end)

    def process_variable(self, GCM, scenario, var, forcing_fp, start, end):

        # create the remapped file name
        remapped_fp = f"gis_{var}_{GCM}_{scenario}_{start}-{end}.nc"

        # remap the forcing file
        self.remap_variable(forcing_fp,
                            remapped_fp,
                            self.test_case.ismip6_2_mali_weights)

        # Now that forcing has been regridded onto the MALI grid, we can drop
        # ISMIP6 grid variables. Special note: the char field `mapping`
        # particularily causes problem with `xtime` (40byte vs. 64byte chars)
        vars_2_drop = ["time_bounds", "lat_vertices", "lon_vertices", "area",
                       "mapping", "lat", "lon"]

        # open the remapped file for post processing
        remapped_ds = xr.open_dataset(remapped_fp,
                                      drop_variables=vars_2_drop,
                                      use_cftime=True)

        # convert time to xtime
        remapped_ds["xtime"] = datetime_2_xtime(remapped_ds, var="time")

        # rename the variable/dimensions to match MPAS/MALI conventions
        remapped_ds = remapped_ds.rename({"ncol": "nCells",
                                          "time": "Time",
                                          var: renaming_dict[var]})

        # SMB is not a var in ISMIP6 forcing files, need to use `SMB_ref`
        # and the processed `aSMB` to produce a `SMB` forcing
        if var == "aSMB":
            # open the reference climatology file
            smb_ref = xr.open_dataset(self.test_case.smb_ref_climatology)
            # squeeze the empty time dimension so that broadcasting works
            smb_ref = smb_ref.squeeze("Time")
            # add climatology to anomoly to make full forcing field
            remapped_ds[renaming_dict[var]] += smb_ref["sfcMassBal"]

        # need to process surfaceAirTemperature
        if var == "surfaceAirTemperature":
            pass

        # write_netcdf(remapped_ds, remapped_fp)
        return remapped_ds

    def remap_variable(self, input_file, output_file, weights_file):
        """
        """

        # remap the forcing file onto the MALI mesh
        args = ["ncremap",
                "-i", input_file,
                "-o", output_file,
                "-m", weights_file]

        check_call(args, logger=self.logger)

    def rename_variable_and_trim_dataset(self, ds, var_src, var_dest):

        # drop unnecessary variables
        ds = ds.drop_vars(["lat_vertices", "area", "lon_vertices",
                           "lat", "lon"])

        ds = ds.rename({"ncol": "nCells",
                        "time": "Time",
                        var_src: var_dest})

        # need to make `Time` the unlimited dimension, which also prevents
        # `time` dimension for being added back to the dataset
        # ds.encoding["unlimited_dims"] = {"Time"}

        return ds

    def create_xtime(self, ds):

        ds["xtime"] = datetime_2_xtime(ds, var="Time")

        return ds
