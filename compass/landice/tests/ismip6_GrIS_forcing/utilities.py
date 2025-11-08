import xarray as xr
from mpas_tools.logging import check_call


def add_xtime(ds, var="Time"):
    """
    Parse DateTime objects and convert to MPAS "xtime" format

    Parameters
    ----------
    ds: xr.Dataset
        Dataset containing the "var" data array of DateTime objects

    var: str
        Name of data array, composed of DateTime objects, to parse

    Returns
    -------
    xtime: xr.DataArray
        DateTime objects converted to 64 byte character strings
    """

    def _datetime_2_xtime(ds, var="Time", xtime_fmt="%Y-%m-%d_%H:%M:%S"):

        def f(t):
            return t.strftime(xtime_fmt).ljust(64)

        return xr.apply_ufunc(f, ds[var], vectorize=True, output_dtypes=["S"])

    # ensure time variable has been properly parsed as a datetime object
    if not hasattr(ds[var], "dt"):
        msg = (
            f"The {var} variable passed has not been parsed as a datetime "
            f"object, so conversion to xtime string will not work.\n\n"
            f"Try using ds = xr.open_dataset(..., use_cftime=True)."
        )
        raise TypeError(msg)

    # xtime related attributes
    attrs = {
        "units": "unitless",
        "long_name": "model time, with format \'YYYY-MM-DD_HH:MM:SS\'"
    }

    # compute xtime dataarray from time variable passed
    xtime = _datetime_2_xtime(ds, var=var)

    # update attributes of dataarray
    xtime.attrs = attrs

    return xtime


def remap_variables(in_fp, out_fp, weights_fp, variables=None, logger=None):
    """
    Remap field using ncremp

    Parameters
    ----------
    in_fp: str
        File path to netcdf that has vars. on source grid

    out_fp: str
        File path to netcdf where the vars. on destination grid will be written

    weight_fp: ste
        File path to the weights file for remapping

    variables: list[str], optional
        List of variable names (as strings) to be remapped.

    logger : logging.Logger
        A logger for capturing the output from ncremap call
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
