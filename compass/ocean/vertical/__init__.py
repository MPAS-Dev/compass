import numpy
import xarray

from compass.ocean.vertical.zlevel import init_z_level_vertical_coord
from compass.ocean.vertical.zstar import init_z_star_vertical_coord


def init_vertical_coord(config, ds):
    """
    Create a vertical coordinate based on the config options in the
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

    for var in ['bottomDepth', 'ssh']:
        if var not in ds:
            raise ValueError('{} must be added to ds before this call.'.format(
                var))

    if 'Time' in ds.ssh.dims:
        # drop it for now, we'll add it back at the end
        ds['ssh'] = ds.ssh.isel(Time=0)

    coord_type = config.get('vertical_grid', 'coord_type')

    if coord_type == 'z-level':
        init_z_level_vertical_coord(config, ds)
    elif coord_type == 'z-star':
        init_z_star_vertical_coord(config, ds)
    elif coord_type == 'haney-number':
        raise ValueError('Haney Number coordinate not yet supported.')
    else:
        raise ValueError('Unknown coordinate type {}'.format(coord_type))

    # recompute the cell mask since min/max indices may have changed
    ds['cellMask'] = _compute_cell_mask(ds.minLevelCell, ds.maxLevelCell,
                                        ds.sizes['nVertLevels'])

    # mask layerThickness and restingThickness
    ds['layerThickness'] = ds.layerThickness.where(ds.cellMask)
    ds['restingThickness'] = ds.restingThickness.where(ds.cellMask)

    # add (back) Time dimension
    ds['ssh'] = ds.ssh.expand_dims(dim='Time', axis=0)
    ds['layerThickness'] = ds.layerThickness.expand_dims(dim='Time', axis=0)
    ds['restingThickness'] = \
        ds.restingThickness.expand_dims(dim='Time', axis=0)

    ds['zMid'] = _compute_zmid_from_layer_thickness(
        ds.layerThickness, ds.ssh, ds.cellMask)

    # fortran 1-based indexing
    ds['minLevelCell'] = ds.minLevelCell+1
    ds['maxLevelCell'] = ds.maxLevelCell+1


def _compute_cell_mask(minLevelCell, maxLevelCell, nVertLevels):
    cellMask = []
    for zIndex in range(nVertLevels):
        mask = numpy.logical_and(zIndex >= minLevelCell,
                                 zIndex <= maxLevelCell)
        cellMask.append(mask)
    cellMask = xarray.concat(cellMask, dim='nVertLevels')
    cellMask = cellMask.transpose('nCells', 'nVertLevels')
    return cellMask


def _compute_zmid_from_layer_thickness(layerThickness, ssh, cellMask):
    """
    Compute zMid from ssh and layerThickness for any vertical coordinate

    Parameters
    ----------
    layerThickness : xarray.DataArray
        The thickness of each layer

    ssh : xarray.DataArray
        The sea surface height

    cellMask : xarray.DataArray
        A boolean mask of where there are valid cells

    Returns
    -------
    zMid : xarray.DataArray
        The elevation of layer centers
    """

    zTop = ssh.copy()
    nVertLevels = layerThickness.sizes['nVertLevels']
    zMid = []
    for zIndex in range(nVertLevels):
        mask = cellMask.isel(nVertLevels=zIndex)
        thickness = layerThickness.isel(nVertLevels=zIndex).where(mask, 0.)
        z = (zTop - 0.5*thickness).where(mask)
        zMid.append(z)
        zTop -= thickness
    zMid = xarray.concat(zMid, dim='nVertLevels').transpose('Time', 'nCells',
                                                            'nVertLevels')
    return zMid
