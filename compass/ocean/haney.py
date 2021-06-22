import xarray
import numpy
import progressbar


def compute_haney_number(ds_mesh, layer_thickness, ssh, show_progress=False):
    """
    Compute the Haney number rx1 for each edge, and interpolate it to cells

    Parameters
    ----------
    ds_mesh : xarray.Dataset
        A dataset with the MPAS-Ocean mesh

    layer_thickness : xarray.DataArray
        A data array with layer thicknesses

    ssh : xarray.DataArray
        A data array with sea surface height

    show_progress : bool, optional
        Whether to show a progress bar

    Returns
    -------
    haney_edge : xarray.DataArray
        A data array with the Haney number at edges and layer interfaces

    haney_cell : xarray.DataArray
        A data array with the Haney number interpolated to cell centers and
        layer interfaces
    """

    nEdges = ds_mesh.sizes['nEdges']
    nCells = ds_mesh.sizes['nCells']
    nVertLevels = ds_mesh.sizes['nVertLevels']
    if 'Time' in layer_thickness.dims:
        nTime = layer_thickness.sizes['Time']
    else:
        nTime = 1
        show_progress = False

    cellsOnEdge = ds_mesh.cellsOnEdge - 1
    minLevelCell = ds_mesh.minLevelCell - 1
    maxLevelCell = ds_mesh.maxLevelCell - 1
    edgesOnCell = ds_mesh.edgesOnCell - 1

    internal_mask = numpy.logical_and(cellsOnEdge[:, 0] >= 0,
                                      cellsOnEdge[:, 1] >= 1)

    cell0 = cellsOnEdge[:, 0]
    cell1 = cellsOnEdge[:, 1]

    minLevelEdge = minLevelCell[cell0]
    mask = numpy.logical_or(cell0 == -1,
                            minLevelCell[cell1] > minLevelEdge)
    minLevelEdge[mask] = minLevelCell[cell1][mask]

    maxLevelEdge = maxLevelCell[cell0]
    mask = numpy.logical_or(cell0 == -1,
                            maxLevelCell[cell1] < maxLevelEdge)
    maxLevelEdge[mask] = maxLevelCell[cell1][mask]

    vert_index = \
        xarray.DataArray.from_dict({'dims': ('nVertLevels',),
                                    'data': numpy.arange(nVertLevels)})

    cell_mask = numpy.logical_and(vert_index >= minLevelCell,
                                  vert_index <= maxLevelCell)

    edge_mask = numpy.logical_and(vert_index >= minLevelEdge,
                                  vert_index <= maxLevelEdge)

    cell0 = cell0[internal_mask]
    cell1 = cell1[internal_mask]

    haney_edge = xarray.DataArray(numpy.zeros((nTime, nEdges, nVertLevels)),
                                  dims=('Time', 'nEdges', 'nVertLevels'))

    haney_cell = xarray.DataArray(numpy.zeros((nTime, nCells, nVertLevels)),
                                  dims=('Time', 'nCells', 'nVertLevels'))

    if show_progress:
        widgets = ['Haney number: ', progressbar.Percentage(), ' ',
                   progressbar.Bar(), ' ', progressbar.ETA()]
        bar = progressbar.ProgressBar(widgets=widgets,
                                      maxval=nTime).start()
    else:
        bar = None

    for tIndex in range(nTime):

        z_mid = numpy.zeros((nCells, nVertLevels+1))
        if 'Time' in layer_thickness.dims:
            local_thickness = layer_thickness.isel(Time=tIndex)
        else:
            local_thickness = layer_thickness
        local_thickness = local_thickness.where(cell_mask, 0.).values
        if 'Time' in ssh.dims:
            local_ssh = ssh.isel(Time=tIndex)
        else:
            local_ssh = ssh
        local_ssh = local_ssh.values
        bottom_depth = ds_mesh.bottomDepth.values
        z_bot = -bottom_depth
        for zIndex in range(nVertLevels-1, -1, -1):
            z_mid[:, zIndex+1] = z_bot + 0.5*local_thickness[:, zIndex]
            z_bot += local_thickness[:, zIndex]
        z_mid[:, 0] = local_ssh

        dz_vert1 = z_mid[cell0, 0:-1] - z_mid[cell0, 1:]
        dz_vert2 = z_mid[cell1, 0:-1] - z_mid[cell1, 1:]
        dz_edge = z_mid[cell1, :] - z_mid[cell0, :]

        dz_vert1[:, 0] *= 2
        dz_vert2[:, 0] *= 2

        rx1 = numpy.zeros((nEdges, nVertLevels))

        epsilon = 1e-10
        denom = dz_vert1 + dz_vert2
        denom[numpy.abs(denom) < epsilon] = epsilon

        rx1[internal_mask, :] = (numpy.abs(dz_edge[:, 0:-1] + dz_edge[:, 1:]) /
                                 denom)

        haney_edge[tIndex, :, :] = rx1
        haney_edge[tIndex, :, :] = haney_edge[tIndex, :, :].where(edge_mask)
        haney_cell[tIndex, :, :] = haney_edge[tIndex, edgesOnCell, :].max(
            dim='maxEdges')
        if show_progress:
            bar.update(tIndex+1)
    if show_progress:
        bar.finish()

    if 'Time' not in layer_thickness.dims:
        # don't need the time dimension
        haney_edge = haney_edge.isel(Time=0)
        haney_cell = haney_cell.isel(Time=0)

    return haney_edge, haney_cell
