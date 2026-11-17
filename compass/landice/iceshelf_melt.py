import numpy as np
from netCDF4 import Dataset, chartostring


def calc_mean_TF(geometry_file, forcing_file, start_year=None,
                 end_year=None):
    """
    Function to calculate mean thermal forcing across all floating ice in a
    domain.

    Parameters
    ----------
    geometry_file : str
        name of geometry file that includes thickness and bedTopography fields

    forcing_file : str
        name of file containing MALI ocean forcing information

    start_year : int, optional
        First year (inclusive) of the period over which the 3D thermal
        forcing is time-averaged before interpolation to the ice draft.
        Years are read from ``xtime`` in ``forcing_file``.  If neither
        ``start_year`` nor ``end_year`` is given, the first time slice is
        used.

    end_year : int, optional
        Last year (inclusive) of the averaging period
    """

    rhoi = 910.0
    rhosw = 1028.0

    ff = Dataset(forcing_file, 'r')
    zOcean = ff.variables['ismip6shelfMelt_zOcean'][:]
    time_indices = _get_time_indices(ff, forcing_file, start_year, end_year)
    TFocean = np.ma.mean(
        ff.variables['ismip6shelfMelt_3dThermalForcing'][time_indices, :, :],
        axis=0)

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


def _get_time_indices(ff, forcing_file, start_year, end_year):
    """
    Get the time indices in the forcing file to average over
    """
    if start_year is None and end_year is None:
        return [0]
    if start_year is None or end_year is None:
        raise ValueError('Both start_year and end_year must be given to '
                         'average thermal forcing over a period.')
    if start_year > end_year:
        raise ValueError(f'start_year ({start_year}) is after end_year '
                         f'({end_year}).')
    if 'xtime' not in ff.variables:
        raise ValueError(f'{forcing_file} has no xtime variable, so a '
                         f'calibration year range cannot be used.')

    xtime = ff.variables['xtime'][:]
    if xtime.ndim > 1:
        xtime = chartostring(xtime)
    years = np.array([int(str(time).strip()[0:4]) for time in xtime])
    time_indices = np.nonzero((years >= start_year) &
                              (years <= end_year))[0]
    if len(time_indices) == 0:
        raise ValueError(f'No time slices in {forcing_file} fall between '
                         f'{start_year} and {end_year}. Available years: '
                         f'{years.min()}-{years.max()}.')
    return time_indices
