import numpy as np
import mpas_tools.mesh.creation.mesh_definition_tools as mdt


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
    dlat = 0.1
    nlon = int(360./dlon) + 1
    nlat = int(180./dlat) + 1
    lon = np.linspace(-180., 180., nlon)
    lat = np.linspace(-90., 90., nlat)

    cellWidthVsLat = mdt.EC_CellWidthVsLat(lat)
    cellWidth = np.outer(cellWidthVsLat, np.ones([1, lon.size]))

    return cellWidth, lon, lat
