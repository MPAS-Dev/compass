import xarray


def compute_layer_thickness_and_zmid(cellMask, refBottomDepth, bottomDepth,
                                     maxLevelCell, ssh=None):
    """
    Initialize the vertical coordinate to a z-star coordinate

    Parameters
    ----------
    cellMask : xarray.DataArray
        A bool mask indicating where cells are valid (above the bathymetry)

    refBottomDepth : xarray.DataArray
        The (positive down) depth of the bottom of each level in a 1D reference
        depth coordinate used for MPAS's z-star coordinate

    bottomDepth : xarray.DataArray
        The (positive down) depth of the bathymetry for each cell in the mesh

    maxLevelCell : xarray.DataArray
        The zero-based index of the last valid level in each cell in the mesh

    ssh : xarray.DataArray, optional
        The sea surface height for each cell in the mesh, assumed to be all
        zeros if not supplied

    Returns
    -------
    restingThickness : xarray.DataArray
        A reference thickness of each layer (level) for all cells and levels in
        the mesh if ``ssh`` were zero everywhere, the same as ``layerThickness``
        if ``ssh`` is not provided

    layerThickness : xarray.DataArray
        The thickness of each layer (level) for all cells and levels in the mesh

    zMid : xarray.DataArray
        The vertical location of the middle of each level for all cells and
        levels in the mesh
    """

    nVertLevels = cellMask.sizes['nVertLevels']

    refLayerThickness = refBottomDepth.isel(nVertLevels=0)

    restingThicknesses = [cellMask.isel(nVertLevels=0) * refLayerThickness]
    for levelIndex in range(1, nVertLevels):
        refLayerThickness = (refBottomDepth.isel(nVertLevels=levelIndex) -
                             refBottomDepth.isel(nVertLevels=levelIndex-1))
        sliceThickness = cellMask.isel(nVertLevels=levelIndex)*refLayerThickness
        mask = levelIndex == maxLevelCell
        partialThickness = (bottomDepth -
                            refBottomDepth.isel(nVertLevels=levelIndex-1))
        sliceThickness = xarray.where(mask, partialThickness, sliceThickness)
        sliceThickness = sliceThickness.where(
            cellMask.isel(nVertLevels=levelIndex))
        restingThicknesses.append(sliceThickness)

    restingThickness = xarray.concat(restingThicknesses, dim='nVertLevels')
    restingThickness = restingThickness.transpose('nCells', 'nVertLevels')

    if ssh is not None:
        layerStretch = (ssh + bottomDepth) / bottomDepth
        layerThickness = restingThickness * layerStretch
    else:
        ssh = xarray.zeros_like(bottomDepth)
        layerThickness = restingThickness

    zBot = ssh - layerThickness.cumsum(dim='nVertLevels')

    zMid = zBot + 0.5*layerThickness

    return restingThickness, layerThickness, zMid
