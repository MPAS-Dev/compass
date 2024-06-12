import xarray as xr
from mpas_tools.logging import check_call


def _datetime_2_xtime(ds, var="Time", xtime_fmt="%Y-%m-%d_%H:%M:%S"):

    return xr.apply_ufunc(lambda t: t.strftime(xtime_fmt).ljust(64),
                          ds[var], vectorize=True, output_dtypes=["S"])


def add_xtime(ds, var="Time"):
    """
    ds["xtime"] = add_xtime(ds)
    """

    # ensure time variable has been properly parsed as a datetime object
    if not hasattr(ds[var], "dt"):
        raise TypeError(f"The {var} variable passed has not been parsed as"
                        " a datetime object, so conversion to xtime string"
                        " will not work.\n\nTry using ds = xr.open_dataset"
                        "(..., use_cftime=True).")

    # xtime related attributes
    attrs = {"units": "unitless",
             "long_name": "model time, with format \'YYYY-MM-DD_HH:MM:SS\'"}

    # compute xtime dataarray from time variable passed
    xtime = _datetime_2_xtime(ds, var=var)

    # update attributes of dataarray
    xtime.attrs = attrs

    return xtime


def remap_variables(in_fp, out_fp, weights_fp, variables=None, logger=None):
    """
    """

    args = ["ncremap",
            "-i", in_fp,
            "-o", out_fp,
            "-m", weights_fp]

    if variables and not isinstance(variables, list):
        raise TypeError("`variables` kwarg must be a list of strings.")
    elif variables:
        args += ["-v"] + variables

    check_call(args, logger=logger)
