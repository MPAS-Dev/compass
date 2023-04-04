import numpy
import numpy as np
from scipy.optimize import root_scalar


def create_index_tanh_dz_grid(num_vert_levels, bottom_depth,
                              min_layer_thickness, max_layer_thickness,
                              transition_levels):
    """
    Creates layer thicknesses that vary as a tanh function in vertical index
    (as opposed to z for the ``tanh_dz`` profile) for MPAS-Ocean

    Parameters
    ----------
    num_vert_levels : int
        Number of vertical levels for the grid

    bottom_depth : float
        bottom depth for the chosen vertical coordinate [m]

    min_layer_thickness : float
        Target thickness of the first layer [m]

    max_layer_thickness : float
        Target maximum thickness in column [m]

    transition_levels : float
        The width of the transition in resolution in layers

    Returns
    -------
    interfaces : numpy.ndarray
        A 1D array of positive depths for layer interfaces in meters
    """

    nz = num_vert_levels
    dz1 = min_layer_thickness
    dz2 = max_layer_thickness
    delta = transition_levels

    bracket = [0, num_vert_levels]

    # the bracket here is large enough that it should hopefully encompass any
    # reasonable value of delta, the characteristic length scale over which
    # dz varies.  The args are passed on to the match_bottom function below,
    # and the root finder will determine a value of delta (sol.root) such that
    # match_bottom is within a tolerance of zero, meaning the bottom of the
    # coordinate computed by cumsum_z hits bottom_depth almost exactly
    sol = root_scalar(_index_tanh_match_bottom, method='brentq',
                      bracket=bracket,
                      args=(nz, dz1, dz2, bottom_depth, delta))

    origin = sol.root
    layerThickness, z = _index_tanh_cumsum_z(delta, nz, dz1, dz2, origin)
    interfaces = -z

    return interfaces


def _index_tanh_match_bottom(origin, nz, dz1, dz2, bottom_depth, delta):
    """
    For tanh layer thickness, compute the difference between the
    bottom depth computed with the given
    parameters and the target ``bottom_depth``, used in the root finding
    algorithm to determine which value of ``delta`` to use.

    Parameters
    ----------
    origin : float
        The layer index around which resolution transitions from the min to the
        max layer thickness (this parameter will be optimized to hit a target
        depth in a target numer of layers)

    nz : int
        The number of layers

    dz1 : float
        The layer thickness at the top of the ocean (z = 0)

    dz2 : float
        The layer thickness at z --> -infinity

    bottom_depth: float
        depth of the bottom of the ocean that should match the bottom layer
        interface.  Note: the bottom_depth is positive, whereas the layer
        interfaces are negative.

    delta : float
        The characteristic number of layers over which dz varies

    Returns
    -------
    diff : float
        The computed bottom depth minus the target ``bottom_depth``.  ``diff``
        should be zero when we have found the desired ``delta``.
    """
    _, z = _index_tanh_cumsum_z(delta, nz, dz1, dz2, origin)
    diff = -bottom_depth - z[-1]
    return diff


def _index_tanh_cumsum_z(delta, nz, dz1, dz2, origin):
    """
    Compute tanh layer interface depths and layer thicknesses over ``nz``
    layers

    Parameters
    ----------
    delta : float
        The characteristic number of layers over which dz varies

    nz : int
        The number of layers

    dz1 : float
        The layer thickness at the top of the ocean (z = 0)

    dz2 : float
        The layer thickness at z --> -infinity

    origin : float
        The layer index around which resolution transitions from the min to the
        max layer thickness (this parameter will be optimized to hit a target
        depth in a target numer of layers)


    Returns
    -------
    dz : numpy.ndarray
        The layer thicknesses for each layer

    z : numpy.ndarray
        The depth (positive up) of each layer interface (``nz + 1`` total
        elements)
    """
    dz = np.zeros(nz)
    z = np.zeros(nz + 1)
    for zindex in range(nz):
        dz[zindex] = _index_tanh_dz_z(zindex, dz1, dz2, delta, origin)
        z[zindex + 1] = z[zindex] - dz[zindex]
    return dz, z


def _index_tanh_dz_z(zindex, dz1, dz2, delta, origin):
    """
    Tanh layer thickness as a function of depth

    Parameters
    ----------
    zindex : float
        The layer index at which to find the layer thickness

    dz1 : float
        The layer thickness at the top of the ocean (z = 0)

    dz2 : float
        The layer thickness at z --> -infinity

    delta : float
        The characteristic number of layers over which dz varies

    origin : float
        The layer index around which resolution transitions from the min to the
        max layer thickness (this parameter will be optimized to hit a target
        depth in a target numer of layers)


    Returns
    -------
    dz : float
        The layer thickness
    """
    tanh = np.tanh((zindex - origin) * np.pi / delta)
    # rescale such that tanh hits zero at index 0
    tanh0 = np.tanh(-origin * np.pi / delta)
    tanh = (tanh - tanh0) / (1. - tanh0)
    return (dz2 - dz1) * tanh + dz1
