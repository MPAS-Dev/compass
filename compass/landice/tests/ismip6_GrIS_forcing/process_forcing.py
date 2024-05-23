import os

import xarray as xr
from mpas_tools.logging import check_call

from compass.step import Step


def datetime_2_xtime(ds, var="time"):
    return xr.apply_ufunc(lambda t: t.strftime("%Y-%m-%d_%H:%M:%S").ljust(64),
                          ds[var], vectorize=True,
                          output_dtypes=["S"])


{"thermal_forcing": "ismip6_2dThermalForcing",
 "basin_runoff": "ismip6Runoff"}


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

            # loop over ocean variables seperately
            for var in ocean_vars:
                var_fp = self.test_case.findForcingFiles(GCM, scenario, var)

                remapped_fp = f"gis_{var}_{GCM}_{scenario}_{start}-{end}.nc"

                self.remap_variable(var_fp,
                                    remapped_fp,
                                    self.test_case.ismip6_2_mali_weights)

                print(var_fp, os.path.exists(var_fp))

            # loop over atmosphere variables
            for variable in atmosphere_vars:
                pass

    def process_variable(self, variables):
        # variable object be initialized in the setup?

        # remap using the landice framework function

        # open the remapped file for post processing
        # remapped_ds = xr.open_dataset(self.out_fn)

        # if var == "SMB":
        # SMB is not a var in ISMIP6 forcing files, need to use `SMB_ref`
        # and the processed `aSMB` to produce a `SMB` forcing

        # rename varibales
        # self.rename_variables(remapped_ds)

        # convert time to xtime
        # self.create_xtime(remapped_ds)
        pass

    def remap_variable(self, input_file, output_file, weights_file):
        """
        """

        # remap the forcing file onto the MALI mesh
        args = ["ncremap",
                "-i", input_file,
                "-o", output_file,
                "-m", weights_file]

        check_call(args, logger=self.logger)

    def rename_variable_and_trim_dataset(self, ds, var):

        # drop unnecessary variables
        ds = ds.drop_vars(["lat_vertices", "area",
                           "lon_vertices", "lat", "lon"])
        pass

    def create_xtime(self, ds):
        pass
