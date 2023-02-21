import numpy as np
from netCDF4 import Dataset


def calc_mean_TF(geometry_file, forcing_file):
    """
    Function to calculate mean thermal forcing across all floating ice in a
    domain.

    Parameters
    ----------
    geometry_file : str
        name of geometry file that includes thickness and bedTopography fields

    forcing_file : str
        name of file containing MALI ocean forcing information
    """

    rhoi = 910.0
    rhosw = 1028.0

    ff = Dataset(forcing_file, 'r')
    zOcean = ff.variables['ismip6shelfMelt_zOcean'][:]
    TFocean = ff.variables['ismip6shelfMelt_3dThermalForcing'][0, :, :]

    f = Dataset(geometry_file, 'r')
    thickness = f.variables['thickness'][0, :]
    bedTopography = f.variables['bedTopography'][0, :]

    floatMask = ((thickness * rhoi / rhosw + bedTopography) < 0.0) * \
                (thickness > 0.0)
    # Note: next line only works for floating areas
    lowerSurface = -1.0 * rhoi / rhosw * thickness
    nCells = len(f.dimensions['nCells'])
    areaCell = f.variables['areaCell'][:]

    TFdraft = np.zeros((nCells,)) * np.nan
    ind = np.where(floatMask == 1)[0]
    for iCell in ind:
        # Linear interpolation of the thermal forcing on the ice draft depth:
        # Note: flip b/c z is ordered from sfc to deep but interp needs to be
        # increasing
        TFdraft[iCell] = np.interp(lowerSurface[iCell], np.flip(zOcean),
                                   np.flip(TFocean[iCell, :]))
    iceshelf_area = areaCell[ind].sum()
    meanTF = (TFdraft[ind] * areaCell[ind]).sum() / iceshelf_area

    f.close()
    ff.close()

    return meanTF, iceshelf_area
