import xarray
import numpy

from compass.ocean.vertical.grid_1d import add_1d_grid
from compass.ocean.vertical.partial_cells import alter_bottom_depth, alter_ssh


def init_z_level_vertical_coord(config, ds):
    """
    Create a z-level vertical coordinate based on the config options in the
    ``vertical_grid`` section and the ``bottomDepth`` and ``ssh`` variables of
    the mesh data set.

    The following new variables will be added to the data set:

      * ``minLevelCell`` - the index of the top valid layer

      * ``maxLevelCell`` - the index of the bottom valid layer

      * ``cellMask`` - a mask of where cells are valid

      * ``layerThickness`` - the thickness of each layer

      * ``restingThickness`` - the thickness of each layer stretched as if
        ``ssh = 0``

      * ``zMid`` - the elevation of the midpoint of each layer

    So far, all supported coordinates make use of a 1D reference vertical grid.
    The following variables associated with that field are also added to the
    mesh:

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
    config : configparser.ConfigParser
        Configuration options with parameters used to construct the vertical
        grid

    ds : xarray.Dataset
        A data set containing ``bottomDepth`` and ``ssh`` variables used to
        construct the vertical coordinate
    """
    add_1d_grid(config, ds)

    ds['vertCoordMovementWeights'] = xarray.ones_like(ds.refBottomDepth)

    ds['minLevelCell'], ds['maxLevelCell'], ds['cellMask'] = \
        compute_min_max_level_cell(ds.refTopDepth, ds.refBottomDepth, ds.ssh,
                                   ds.bottomDepth)

    ds['bottomDepth'], ds['maxLevelCell'] = alter_bottom_depth(
        config, ds.bottomDepth, ds.refBottomDepth, ds.maxLevelCell)

    ds['ssh'], ds['minLevelCell'] = alter_ssh(
        config, ds.ssh, ds.refBottomDepth, ds.minLevelCell)

    ds['layerThickness'] = compute_z_level_layer_thickness(
        ds.refTopDepth, ds.refBottomDepth, ds.ssh, ds.bottomDepth,
        ds.minLevelCell, ds.maxLevelCell)

    ds['restingThickness'] = compute_z_level_resting_thickness(
        ds.layerThickness, ds.ssh, ds.bottomDepth, ds.minLevelCell,
        ds.maxLevelCell)


def compute_min_max_level_cell(refTopDepth, refBottomDepth, ssh, bottomDepth):
    """
    Compute ``minLevelCell`` and ``maxLevelCell`` indices as well as a cell
    mask for the given reference grid and top and bottom topography.

    Parameters
    ----------
    refTopDepth : xarray.DataArray
        A 1D array of positive-down depths of the top of each z level

    refBottomDepth : xarray.DataArray
        A 1D array of positive-down depths of the bottom of each z level

    ssh : xarray.DataArray
        The sea surface height

    bottomDepth : xarray.DataArray
        The positive-down depth of the seafloor


    Returns
    -------
    minLevelCell : xarray.DataArray
        The zero-based index of the top valid level

    maxLevelCell : xarray.DataArray
        The zero-based index of the bottom valid level

    cellMask : xarray.DataArray
        A boolean mask of where there are valid cells
    """
    valid = bottomDepth > -ssh

    aboveTopMask = (refBottomDepth <= -ssh).transpose('nCells', 'nVertLevels')
    aboveBottomMask = (refTopDepth < bottomDepth).transpose(
        'nCells', 'nVertLevels')
    aboveBottomMask = numpy.logical_and(aboveBottomMask, valid)

    minLevelCell = (aboveTopMask.sum(dim='nVertLevels')).where(valid, 0)
    maxLevelCell = (aboveBottomMask.sum(dim='nVertLevels') - 1).where(valid, 0)

    cellMask = numpy.logical_and(numpy.logical_not(aboveTopMask),
                                 aboveBottomMask)
    cellMask = numpy.logical_and(cellMask, valid)

    return minLevelCell, maxLevelCell, cellMask


def compute_z_level_layer_thickness(refTopDepth, refBottomDepth, ssh,
                                    bottomDepth, minLevelCell, maxLevelCell):
    """
    Compute z-level layer thickness from ssh and bottomDepth

    Parameters
    ----------
    refTopDepth : xarray.DataArray
        A 1D array of positive-down depths of the top of each z level

    refBottomDepth : xarray.DataArray
        A 1D array of positive-down depths of the bottom of each z level

    ssh : xarray.DataArray
        The sea surface height

    bottomDepth : xarray.DataArray
        The positive-down depth of the seafloor

    minLevelCell : xarray.DataArray
        The zero-based index of the top valid level

    maxLevelCell : xarray.DataArray
        The zero-based index of the bottom valid level

    Returns
    -------
    layerThickness : xarray.DataArray
        The thickness of each layer (level)
    """

    nVertLevels = refBottomDepth.sizes['nVertLevels']
    layerThickness = []
    for zIndex in range(nVertLevels):
        mask = numpy.logical_and(zIndex >= minLevelCell,
                                 zIndex <= maxLevelCell)
        zTop = numpy.minimum(ssh, -refTopDepth[zIndex])
        zBot = numpy.maximum(-bottomDepth, -refBottomDepth[zIndex])
        thickness = (zTop - zBot).where(mask, 0.)
        layerThickness.append(thickness)
    layerThickness = xarray.concat(layerThickness, dim='nVertLevels')
    layerThickness = layerThickness.transpose('nCells', 'nVertLevels')
    return layerThickness


def compute_z_level_resting_thickness(layerThickness, ssh, bottomDepth,
                                      minLevelCell, maxLevelCell):
    """
    Compute z-level resting thickness by "unstretching" layerThickness
    based on ssh and bottomDepth

    Parameters
    ----------
    layerThickness : xarray.DataArray
        The thickness of each layer (level)

    ssh : xarray.DataArray
        The sea surface height

    bottomDepth : xarray.DataArray
        The positive-down depth of the seafloor

    minLevelCell : xarray.DataArray
        The zero-based index of the top valid level

    maxLevelCell : xarray.DataArray
        The zero-based index of the bottom valid level

    Returns
    -------
    restingThickness : xarray.DataArray
        The thickness of z-star layers when ssh = 0
    """

    nVertLevels = layerThickness.sizes['nVertLevels']
    restingThickness = []

    layerStretch = bottomDepth / (ssh + bottomDepth)
    for zIndex in range(nVertLevels):
        mask = numpy.logical_and(zIndex >= minLevelCell,
                                 zIndex <= maxLevelCell)
        thickness = layerStretch * layerThickness.isel(
            nVertLevels=zIndex)
        thickness = thickness.where(mask, 0.)
        restingThickness.append(thickness)
    restingThickness = xarray.concat(restingThickness, dim='nVertLevels')
    restingThickness = restingThickness.transpose('nCells', 'nVertLevels')
    return restingThickness
