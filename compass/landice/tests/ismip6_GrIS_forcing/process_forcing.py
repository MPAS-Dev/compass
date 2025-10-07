import os

import xarray as xr
from mpas_tools.io import write_netcdf

from compass.landice.tests.ismip6_GrIS_forcing.utilities import (
    add_xtime,
    remap_variables,
)
from compass.step import Step

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

            atm_fn = f"gis_atm_forcing_{GCM}_{scenario}_{start}--{end}.nc"
            self.process_variables(GCM, scenario, start, end, atmosphere_vars,
                                   atm_fn)

            ocn_fn = f"gis_ocn_forcing_{GCM}_{scenario}_{start}--{end}.nc"
            self.process_variables(GCM, scenario, start, end, ocean_vars,
                                   ocn_fn)

    def process_variables(self, GCM, scenario, start, end,
                          variables, output_fn):

        forcing_datastes = []

        for var in variables:
            var_fp = self.test_case.findForcingFiles(GCM, scenario, var)

            ds = self.process_variable(GCM, scenario, var, var_fp, start, end)

            forcing_datastes.append(ds)

        forcing_ds = xr.merge(forcing_datastes)

        write_netcdf(forcing_ds, output_fn)

    def process_variable(self, GCM, scenario, var, forcing_fp, start, end):

        #
        config = self.config
        #
        renamed_var = renaming_dict[var]

        # create the remapped file name, using original variable name
        remapped_fn = f"remapped_{GCM}_{scenario}_{var}_{start}-{end}.nc"
        # follow the directory structure of the concatenated source files
        remapped_fp = os.path.join(f"{GCM}-{scenario}/{var}", remapped_fn)

        # remap the forcing file
        remap_variables(forcing_fp,
                        remapped_fp,
                        self.test_case.ismip6_2_mali_weights,
                        variables=[var],
                        logger=self.logger)

        # Now that forcing has been regridded onto the MALI grid, we can drop
        # ISMIP6 grid variables. Special note: the char field `mapping`
        # particularily causes problem with `xtime` (40byte vs. 64byte chars)
        vars_2_drop = ["time_bounds", "lat_vertices", "lon_vertices", "area",
                       "mapping", "lat", "lon", "polar_stereographic"]

        # open the remapped file for post processing
        remapped_ds = xr.open_dataset(remapped_fp,
                                      drop_variables=vars_2_drop,
                                      use_cftime=True)

        # create mask of desired time indices. Include forcing from year prior
        # to requested start date since forcing in July but sims start in Jan
        mask = remapped_ds.time.dt.year.isin(range(start - 1, end))
        # drop the non-desired time indices from remapped dataset
        remapped_ds = remapped_ds.where(mask, drop=True)
        # add xtime variable based on `time`
        remapped_ds["xtime"] = add_xtime(remapped_ds, var="time")

        # rename the variable/dimensions to match MPAS/MALI conventions
        remapped_ds = remapped_ds.rename({"ncol": "nCells",
                                          "time": "Time",
                                          var: renamed_var})

        # drop the unneeded attributes from the dataset
        for _var in remapped_ds:
            remapped_ds[_var].attrs.pop("grid_mapping", None)
            remapped_ds[_var].attrs.pop("cell_measures", None)

        # SMB is not a var in ISMIP6 forcing files, need to use `SMB_ref`
        # and the processed `aSMB` to produce a `SMB` forcing
        if renamed_var == "sfcMassBal":
            # open the reference climatology file
            smb_ref = xr.open_dataset(self.test_case.smb_ref_climatology)
            # squeeze the empty time dimension so that broadcasting works
            smb_ref = smb_ref.squeeze("Time")
            # add climatology to anomoly to make full forcing field
            remapped_ds[renamed_var] += smb_ref[renamed_var]

        if renamed_var == "surfaceAirTemperature":
            # read the mesh path/name from the config file
            mali_mesh_fp = config.get("ISMIP6_GrIS_Forcing", "MALI_mesh_fp")
            # squeeze the empty time dimension so that broadcasting works
            mali_ds = xr.open_dataset(mali_mesh_fp).squeeze("Time")

            remapped_ds[renamed_var] += mali_ds[renamed_var]

        # ocean variable contain alot of nan's, replace with zeros
        # ismip6Runoff : nan in catchments that are not marine terminating
        # ismip6_2dThermalForcing : nan for all cells above sea level
        if renamed_var in {"ismip6_2dThermalForcing", "ismip6Runoff"}:
            # get the ocean dataarray of interest
            da = remapped_ds[renamed_var]
            # set nan values to zero in the parent dataset
            remapped_ds[renamed_var] = xr.where(da.isnull(), 0, da)

        return remapped_ds
