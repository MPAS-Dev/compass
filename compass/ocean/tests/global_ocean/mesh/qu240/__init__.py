import numpy as np


def build_cell_width_lat_lon():
    """
    Create cell width array for this mesh on a regular latitude-longitude grid

    Returns
    -------
    cellWidth : numpy.array
        m x n array of cell width in km

    lon : numpy.array
        longitude in degrees (length n and between -180 and 180)

    lat : numpy.array
        longitude in degrees (length m and between -90 and 90)
    """
    dlon = 10.
    dlat = dlon
    constantCellWidth = 240.

    nlat = int(180/dlat) + 1
    nlon = int(360/dlon) + 1
    lat = np.linspace(-90., 90., nlat)
    lon = np.linspace(-180., 180., nlon)

    cellWidth = constantCellWidth * np.ones((lat.size, lon.size))
    return cellWidth, lon, lat
