import xarray

from compass.ocean.vertical.grid_1d import add_1d_grid


def init_sigma_vertical_coord(config, ds):
    """
    Create a sigma (terrain-following) vertical coordinate based on the config
    options in the `vertical_grid`` section and the ``bottomDepth`` and
    ``ssh`` variables of the mesh data set.

    The following new variables will be added to the data set:

      * ``minLevelCell`` - the index of the top valid layer

      * ``maxLevelCell`` - the index of the bottom valid layer

      * ``cellMask`` - a mask of where cells are valid

      * ``layerThickness`` - the thickness of each layer

      * ``restingThickness`` - the thickness of each layer stretched as if
        ``ssh = 0``

      * ``zMid`` - the elevation of the midpoint of each layer

    The sigma coordinate makes use of a 1D reference vertical grid. The
    following variables associated with that field are also added to the mesh:

      * ``refTopDepth`` - the positive-down depth of the top of each ref. level

      * ``refZMid`` - the positive-down depth of the middle of each ref. level

      * ``refBottomDepth`` - the positive-down depth of the bottom of each ref.
        level

      * ``refInterfaces`` - the positive-down depth of the interfaces between
        ref. levels (with ``nVertLevels`` + 1 elements).

      * ``vertCoordMovementWeights`` - the weights (all ones) for coordinate
        movement

    There is considerable redundancy between these variables but each is
    sometimes convenient.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options with parameters used to construct the vertical
        grid

    ds : xarray.Dataset
        A data set containing ``bottomDepth`` and ``ssh`` variables used to
        construct the vertical coordinate
    """
    add_1d_grid(config, ds)

    ds['vertCoordMovementWeights'] = xarray.ones_like(ds.refBottomDepth)

    valid = ds.bottomDepth > -ds.ssh
    cell_mask, _ = xarray.broadcast(valid, ds.refBottomDepth)
    ds['cellMask'] = cell_mask.transpose('nCells', 'nVertLevels')

    ds['minLevelCell'] = xarray.zeros_like(ds.bottomDepth, dtype=int)
    ds['maxLevelCell'] = (ds.sizes['nVertLevels']-1 *
                          xarray.ones_like(ds.bottomDepth, dtype=int))

    resting_ssh = xarray.zeros_like(ds.bottomDepth)

    ds['restingThickness'] = compute_sigma_layer_thickness(
        ds.refInterfaces, resting_ssh, ds.bottomDepth)

    ds['layerThickness'] = compute_sigma_layer_thickness(
        ds.refInterfaces, ds.ssh, ds.bottomDepth)


def compute_sigma_layer_thickness(ref_interfaces, ssh, bottom_depth,
                                  max_level=None):
    """
    Compute sigma layer thickness by stretching restingThickness based on ssh
    and bottom_depth

    Parameters
    ----------
    ref_interfaces : xarray.DataArray
        the interfaces of the reference coordinate used to define sigma layer
        spacing.  The interfaces are renormalized to be between 0 and 1

    ssh : xarray.DataArray
        The sea surface height

    bottom_depth : xarray.DataArray
        The positive-down depth of the seafloor

    max_level : int, optional
        The maximum number of levels used for the sigma coordinate.  The
        default is ``nVertLevels``.

    Returns
    -------
    layer_thickness : xarray.DataArray
        The thickness of each layer
    """
    n_vert_levels = ref_interfaces.sizes['nVertLevelsP1']-1
    if max_level is None:
        max_level = n_vert_levels

    # renormalize ref_interfaces to a coordinate between 0 and 1
    stretch = (ref_interfaces - ref_interfaces.min()).values
    stretch = stretch / stretch[max_level]

    column_thickness = ssh + bottom_depth
    valid = bottom_depth > -ssh
    layer_thickness = []

    for z_index in range(max_level):
        thickness = (stretch[z_index+1] - stretch[z_index])*column_thickness
        thickness = thickness.where(valid, 0.)
        layer_thickness.append(thickness)
    for z_index in range(max_level, n_vert_levels):
        layer_thickness.append(xarray.zeros_like(bottom_depth))
    layer_thickness = xarray.concat(layer_thickness, dim='nVertLevels')
    layer_thickness = layer_thickness.transpose('nCells', 'nVertLevels')
    return layer_thickness
