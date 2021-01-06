#!/usr/bin/env python
import numpy as np
from channel import channel
from jigsaw_to_netcdf_periodic import jigsaw_to_netcdf_periodic

def cellWidthVsLatLon():
    """
    Create cell width array for this mesh on a regular latitude-longitude grid.

    Returns
    -------
    cellWidth : ndarray
        m x n array of cell width in km

    lon : ndarray
        longitude in degrees (length n and between -180 and 180)

    lat : ndarray
        longitude in degrees (length m and between -90 and 90)
    """
    ddeg = 10
# todo: make higher resolution cases, follow Rossby Radius RRS etc.
    constantCellWidth = 100

    lat = np.arange(-90, 90.01, ddeg)
    lon = np.arange(-180, 180.01, ddeg)

    cellWidth = constantCellWidth * np.ones((lat.size, lon.size))
    return cellWidth, lon, lat


def main():
    cellWidth, lon, lat = cellWidthVsLatLon()
    channel()
    msh_filename = 'mesh.msh'
    init_filename = 'init.msh'
    output_name = 'base_mesh.nc'
    on_sphere = True
    SPHERE_RADIUS = +6371.0
    jigsaw_to_netcdf_periodic(msh_filename, init_filename, output_name, on_sphere, sphere_radius=SPHERE_RADIUS)

if __name__ == '__main__':
    main()
