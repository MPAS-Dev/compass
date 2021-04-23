# import new MPAS cores here
from compass.landice import Landice
from compass.ocean import Ocean


def get_mpas_cores():
    """
    Get a list of all collections of tests for MPAS cores

    Returns
    -------
    mpas_cores : list of compass.MpasCore
        A list of MPAS cores containing all available tests
    """
    # add new MPAS cores here
    mpas_cores = [Landice(), Ocean()]
    return mpas_cores
