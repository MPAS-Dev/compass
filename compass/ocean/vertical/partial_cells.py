import xarray
import numpy


def alter_bottom_depth(config, bottomDepth, refBottomDepth, maxLevelCell):
    """
    Alter ``bottomDepth`` and ``maxLevelCell`` for full or partial top cells,
    if requested

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options with parameters used to construct the vertical
        grid

    bottomDepth : xarray.DataArray
        The positive-down depth of the seafloor

    refBottomDepth : xarray.DataArray
        A 1D array of positive-down depths of the bottom of each z level

    maxLevelCell : xarray.DataArray
        The zero-based index of the bottom valid level

    Returns
    -------
    bottomDepth : xarray.DataArray
        The positive-down depth of the seafloor, after alteration

    maxLevelCell : xarray.DataArray
        The zero-based index of the bottom valid level, after alteration
    """
    section = config['vertical_grid']
    partial_cell_type = 'none'
    min_pc_fraction = 0.
    if config.has_option('vertical_grid', 'partial_cell_type'):

        partial_cell_type = section.get('partial_cell_type').lower()
        min_pc_fraction = section.getfloat('min_pc_fraction')

    if partial_cell_type == 'full':
        bottomDepth = _compute_full_cells_depth(
            refBottomDepth, maxLevelCell)
    elif partial_cell_type == 'partial':
        bottomDepth, maxLevelCell = _alter_bottom_depth_for_partial_cells(
            bottomDepth, refBottomDepth, maxLevelCell, min_pc_fraction)
    elif partial_cell_type != 'none':
        raise ValueError('Unexpected partial cell type {}'.format(
            partial_cell_type))

    return bottomDepth, maxLevelCell


def alter_ssh(config, ssh, refBottomDepth, minLevelCell):
    """
    Alter ``ssh`` and ``maxLevelCell`` for full or partial top cells,
    if requested

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options with parameters used to construct the vertical
        grid

    ssh : xarray.DataArray
        The sea surface height

    refBottomDepth : xarray.DataArray
        A 1D array of positive-down depths of the bottom of each z level

    minLevelCell : xarray.DataArray
        The zero-based index of the top valid level

    Returns
    -------
    ssh : xarray.DataArray
        The sea surface height, after alteration

    minLevelCell : xarray.DataArray
        The zero-based index of the top valid level, after alteration
    """
    section = config['vertical_grid']
    partial_cell_type = 'none'
    min_pc_fraction = 0.
    if config.has_option('vertical_grid', 'partial_cell_type'):
        partial_cell_type = section.get('partial_cell_type').lower()
        min_pc_fraction = section.getfloat('min_pc_fraction')

    if partial_cell_type == 'full':
        ssh = _compute_full_cells_depth(
            refBottomDepth, minLevelCell-1)
    elif partial_cell_type == 'partial':
        ssh, minLevelCell = _alter_ssh_for_partial_cells(
            ssh, refBottomDepth, minLevelCell, min_pc_fraction)
    elif partial_cell_type != 'none':
        raise ValueError('Unexpected partial cell type {}'.format(
            partial_cell_type))

    return ssh, minLevelCell


def _compute_full_cells_depth(refBottomDepth, levelIndex):
    """
    Compute the full cell bottom depth given a level index
    """

    depth = refBottomDepth.isel(nVertLevels=levelIndex).where(
        levelIndex >= 0, other=0.)
    return depth


def _alter_bottom_depth_for_partial_cells(bottomDepth, refBottomDepth,
                                          maxLevelCell, min_pc_fraction):
    """
    Alter bottomDepth and maxLevelCell for partial cells
    """

    fullBot = _compute_full_cells_depth(refBottomDepth, maxLevelCell)

    fullTop = _compute_full_cells_depth(refBottomDepth, maxLevelCell-1)

    fullThickness = fullBot - fullTop

    minBottomDepth = fullBot - (1. - min_pc_fraction)*fullThickness

    minBottomDepthMid = 0.5*(minBottomDepth + fullTop)

    # where the bottom depth is far too shallow, we're going to fill in the
    # last level
    mask = bottomDepth < minBottomDepthMid
    maxLevelCell = xarray.where(mask, maxLevelCell-1, maxLevelCell)
    bottomDepth = xarray.where(mask, fullTop, bottomDepth)

    # where the bottom depth only a bit too shallows, we move it deeper
    mask = numpy.logical_and(numpy.logical_not(mask),
                             bottomDepth < minBottomDepth)
    bottomDepth = xarray.where(mask, minBottomDepth, bottomDepth)

    return bottomDepth, maxLevelCell


def _alter_ssh_for_partial_cells(ssh, refBottomDepth, minLevelCell,
                                 min_pc_fraction):
    """
    Alter ssh and minLevelCell for partial cells
    """

    zBot = -_compute_full_cells_depth(refBottomDepth, minLevelCell)

    zTop = -_compute_full_cells_depth(refBottomDepth, minLevelCell-1)

    fullThickness = zTop - zBot

    minSSH = zBot + min_pc_fraction * fullThickness

    minSSHMid = 0.5 * (minSSH + zBot)

    # where the SSH is far too deep, we're going to fill the current top level
    mask = ssh < minSSHMid
    minLevelCell = xarray.where(mask, minLevelCell + 1, minLevelCell)
    ssh = xarray.where(mask, zBot, ssh)

    # where the SSH only a bit too deep, we move it shallower
    mask = numpy.logical_and(numpy.logical_not(mask), ssh < minSSH)
    ssh = xarray.where(mask, minSSH, ssh)

    return ssh, minLevelCell
