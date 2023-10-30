import numpy
import numpy as np


def create_linear_dz_grid(num_vert_levels, bottom_depth,
                          linear_dz_rate):
    """
    Creates the linear vertical grid for MPAS-Ocean and
    writes it to a NetCDF file

    Parameters
    ----------
    num_vert_levels : int
        Number of vertical levels for the grid

    bottom_depth : float
        bottom depth for the chosen vertical coordinate [m]

    linear_dz_rate : float
        rate of layer thickness increase (for linear_dz) [m]

    Returns
    -------
    interfaces : numpy.ndarray
        A 1D array of positive depths for layer interfaces in meters
    """

    nz = num_vert_levels
    layerThickness = [(bottom_depth / nz) - (np.floor(nz / 2) - k) *
                      linear_dz_rate for k in np.arange(0, nz)]
    min_layer_thickness = layerThickness[0]
    max_layer_thickness = layerThickness[-1]
    print('Linear dz vertical grid')
    print(f'min layer thickness: {min_layer_thickness}; '
          f'max layer thickness {max_layer_thickness} in m;')
    interfaces = - np.append([0], np.cumsum(layerThickness))

    return interfaces
