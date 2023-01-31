import xarray
import xarray.plot
import numpy as np


def compute_rpe(initial_state_file_name='initial_state.nc',
                output_file_prefix='output_', num_files=5):
    """
    Computes the reference (resting) potential energy for the whole domain
    Parameters
    ----------
    initial_state_file_name : str
        Name of the netCDF file containing the initial state

    output_file_prefix : str
        Prefix for the netCDF file containing output of forward step

    num_files : int
        Number of files on which to compute rpe

    Returns
    -------
    rpe : numpy.ndarray
          the reference potential energy of size (num_timesteps) x (num_files)
    """
    gravity = 9.80616

    # --- Open and read vars from netCDF file
    dsInit = xarray.open_dataset(initial_state_file_name)
    nCells = dsInit.sizes['nCells']
    nEdges = dsInit.sizes['nEdges']
    nVertLevels = dsInit.sizes['nVertLevels']

    xCell = dsInit.xCell
    yCell = dsInit.yCell
    xEdge = dsInit.xEdge
    yEdge = dsInit.yEdge
    areaCell = dsInit.areaCell
    minLevelCell = dsInit.minLevelCell - 1
    maxLevelCell = dsInit.maxLevelCell - 1
    bottomDepth = dsInit.bottomDepth

    # --- Compute a few quantities relevant to domain geometry
    areaCellMatrix = np.tile(areaCell, (nVertLevels, 1)).transpose()
    bottomMax = np.max(bottomDepth.values)
    yMin = np.min(yEdge.values)
    yMax = np.max(yEdge.values)
    xMin = np.min(xEdge.values)
    xMax = np.max(xEdge.values)
    areaDomain = (yMax - yMin) * (xMax - xMin)

    vert_index = \
        xarray.DataArray.from_dict({'dims': ('nVertLevels',),
                                    'data': np.arange(nVertLevels)})

    cell_mask = np.logical_and(vert_index >= minLevelCell,
                               vert_index <= maxLevelCell)
    cell_mask = np.swapaxes(cell_mask, 0, 1)
    nCells_1D = len(areaCellMatrix[cell_mask])

    # --- Get number of timesteps, assuming all output files have the same
    # --- number of timesteps
    ds = xarray.open_dataset('{}{}.nc'.format(
                             output_file_prefix, 1))
    nt = ds.sizes['Time']
    ds.close()

    # --- Allocations
    rpe1 = np.zeros(nCells_1D)
    zMid = np.zeros(nCells_1D)
    thickness = np.zeros(nCells_1D)
    rpe = np.ones((num_files, nt))

    for n in range(num_files):

        ds = xarray.open_dataset('{}{}.nc'.format(
                                 output_file_prefix, n+1))

        xtime = ds.xtime.values
        hFull = ds.layerThickness
        densityFull = ds.density

        for tidx, t in enumerate(xtime):

            h = hFull[tidx, :, :].values
            vol = np.multiply(h, areaCellMatrix)
            density = densityFull[tidx, :, :].values
            density_1D = density[cell_mask]
            vol_1D = vol[cell_mask]

            # --- Density sorting in ascending order
            sorted_ind = np.argsort(density_1D)
            density_sorted = density_1D[sorted_ind]
            vol_sorted = vol_1D[sorted_ind]

            thickness = np.divide(vol_sorted.tolist(), areaDomain)

            # --- RPE computation
            z = np.append([0.], -np.cumsum(thickness))
            zMid = z[0:-1] - thickness/2. + bottomMax
            rpe1 = gravity * np.multiply(
                       np.multiply(density_sorted, zMid),
                       vol_sorted)

            rpe[n, tidx] = np.sum(rpe1)/np.sum(areaCell)

        ds.close()

    # --- Write rpe to text file
    rows = ['{}'.format(t.astype(str)) for t in xtime]
    with open('rpe.txt', 'w+') as csvfile:
        col_headings = 'time'
        for n in range(num_files):
            col_headings += ',output_' + str(n+1)
        col_headings += '\n'
        csvfile.writelines(col_headings)
        for tidx, t in enumerate(xtime):
            for n in range(num_files):
                rows[tidx] += ',%f' % rpe[n, tidx]
            rows[tidx] += '\n'
        csvfile.writelines(rows)
    csvfile.close()

    return rpe
