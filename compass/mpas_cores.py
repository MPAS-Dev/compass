# import new MPAS cores here
from compass.ocean import Ocean


def get_mpas_cores():
    """

    Returns
    -------
    mpas_cores : list of compass.MpasCore
        A list of MPAS cores containing all available tests

    """
    mpas_cores = list()
    # add new MPAS cores here
    mpas_cores.append(Ocean())
    return mpas_cores
