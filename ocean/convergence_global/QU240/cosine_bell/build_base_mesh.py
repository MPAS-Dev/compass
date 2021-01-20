#!/usr/bin/env python
import numpy as np
from mpas_tools.ocean import build_spherical_mesh


def cellWidthVsLatLon():
    """
    Create cell width array for this mesh on a regular latitude-longitude grid.

    Returns
    -------
    cellWidth : numpy.array
        m x n array, entries are desired cell width in km
    lat : numpy.array
        latitude, vector of length m, with entries between -90 and 90, degrees
    lon : numpy.array
        longitude, vector of length n, with entries between -180 and 180,
        degrees
    """

    ddeg = 10
    constantCellWidth = 240

    lat = np.arange(-90, 90.01, ddeg)
    lon = np.arange(-180, 180.01, ddeg)

    cellWidth = constantCellWidth * np.ones((lat.size, lon.size))
    return cellWidth, lon, lat


def main():
    cellWidth, lon, lat = cellWidthVsLatLon()
    build_spherical_mesh(cellWidth, lon, lat, out_filename='base_mesh.nc')


if __name__ == '__main__':
    main()
