import xarray
import xarray.plot
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def plot_initial_state(input_file_name='initial_state.nc',
                       output_file_name='initial_state.png'):
    """
    creates histogram plots of the initial condition

    Parameters
    ----------
    input_file_name : str, optional
        The path to a NetCDF file with the initial state

    output_file_name: str, optional
        The path to the output image file
    """

    # load mesh variables
    chunks = {'nCells': 32768, 'nEdges': 32768}
    ds = xarray.open_dataset(input_file_name, chunks=chunks)
    nCells = ds.sizes['nCells']
    nEdges = ds.sizes['nEdges']
    nVertLevels = ds.sizes['nVertLevels']

    fig = plt.figure()
    fig.set_size_inches(16.0, 12.0)
    plt.clf()

    print('plotting histograms of the initial condition')
    print('see: init/initial_state/initial_state.png')
    d = datetime.datetime.today()
    txt = \
        'MPAS-Ocean initial state\n' + \
        'date: {}\n'.format(d.strftime('%m/%d/%Y')) + \
        'number cells: {}\n'.format(nCells) + \
        'number cells, millions: {:6.3f}\n'.format(nCells / 1.e6) + \
        'number layers: {}\n\n'.format(nVertLevels) + \
        '  min val   max val  variable name\n'

    plt.subplot(3, 3, 2)
    varName = 'maxLevelCell'
    var = ds[varName]
    maxLevelCell = var.values - 1
    xarray.plot.hist(var, bins=nVertLevels - 4)
    plt.ylabel('frequency')
    plt.xlabel(varName)
    txt = '{}{:9.2e} {:9.2e} {}\n'.format(txt, var.min().values,
                                          var.max().values, varName)

    plt.subplot(3, 3, 3)
    varName = 'bottomDepth'
    var = ds[varName]
    xarray.plot.hist(var, bins=nVertLevels - 4)
    plt.xlabel(varName)
    txt = '{}{:9.2e} {:9.2e} {}\n'.format(txt, var.min().values,
                                          var.max().values, varName)

    cellsOnEdge = ds['cellsOnEdge'].values - 1
    cellMask = np.zeros((nCells, nVertLevels), bool)
    edgeMask = np.zeros((nEdges, nVertLevels), bool)
    for k in range(nVertLevels):
        cellMask[:, k] = k <= maxLevelCell
        cell0 = cellsOnEdge[:, 0]
        cell1 = cellsOnEdge[:, 1]
        edgeMask[:, k] = np.logical_and(np.logical_and(cellMask[cell0, k],
                                                       cellMask[cell1, k]),
                                        np.logical_and(cell0 >= 0,
                                                       cell1 >= 0))
    cellMask = xarray.DataArray(data=cellMask, dims=('nCells', 'nVertLevels'))
    edgeMask = xarray.DataArray(data=edgeMask, dims=('nEdges', 'nVertLevels'))

    plt.subplot(3, 3, 4)
    varName = 'temperature'
    var = ds[varName].isel(Time=0).where(cellMask)
    xarray.plot.hist(var, bins=100, log=True)
    plt.ylabel('frequency')
    plt.xlabel(varName)
    txt = '{}{:9.2e} {:9.2e} {}\n'.format(txt, var.min().values,
                                          var.max().values, varName)

    plt.subplot(3, 3, 5)
    varName = 'salinity'
    var = ds[varName].isel(Time=0).where(cellMask)
    xarray.plot.hist(var, bins=100, log=True)
    plt.xlabel(varName)
    txt = '{}{:9.2e} {:9.2e} {}\n'.format(txt, var.min().values,
                                          var.max().values, varName)

    plt.subplot(3, 3, 6)
    varName = 'layerThickness'
    var = ds[varName].isel(Time=0).where(cellMask)
    xarray.plot.hist(var, bins=100, log=True)
    plt.xlabel(varName)
    txt = '{}{:9.2e} {:9.2e} {}\n'.format(txt, var.min().values,
                                          var.max().values, varName)

    plt.subplot(3, 3, 7)
    varName = 'rx1Edge'
    var = ds[varName].isel(Time=0).where(edgeMask)
    maxRx1Edge = var.max().values
    xarray.plot.hist(var, bins=100, log=True)
    plt.ylabel('frequency')
    plt.xlabel('Haney Number, max={:4.2f}'.format(maxRx1Edge))
    txt = '{}{:9.2e} {:9.2e} {}\n'.format(txt, var.min().values,
                                          var.max().values, varName)

    font = FontProperties()
    font.set_family('monospace')
    font.set_size(12)
    print(txt)
    plt.subplot(3, 3, 1)
    plt.text(0, 1, txt, verticalalignment='top', fontproperties=font)
    plt.axis('off')

    plt.tight_layout(pad=4.0)

    plt.savefig(output_file_name, bbox_inches='tight', pad_inches=0.1)


def plot_vertical_grid(grid_filename, config,
                       out_filename='vertical_grid.png'):
    """
    Plot the vertical grid

    Parameters
    ----------
    grid_filename : str
        The name of the NetCDF file containing the vertical grid

    config : configparser.ConfigParser
        Configuration options for the vertical grid

    out_filename : str, optional
        The name of the image file to write to
    """

    ds = xarray.open_dataset(grid_filename)
    nVertLevels = ds.sizes['nVertLevels']
    midDepth = ds.refMidDepth.values
    layerThickness = ds.refLayerThickness.values
    botDepth = ds.refBottomDepth.values

    fig = plt.figure()
    fig.set_size_inches(16.0, 8.0)
    zInd = np.arange(1, nVertLevels + 1)
    plt.clf()

    plt.subplot(2, 2, 1)
    plt.plot(zInd, midDepth, '.')
    plt.gca().invert_yaxis()
    plt.xlabel('vertical index (one-based)')
    plt.ylabel('layer mid-depth [m]')
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(layerThickness, midDepth, '.')
    plt.gca().invert_yaxis()
    plt.xlabel('layer thickness [m]')
    plt.ylabel('layer mid-depth [m]')
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(zInd, layerThickness, '.')
    plt.xlabel('vertical index (one-based)')
    plt.ylabel('layer thickness [m]')
    plt.grid()

    txt = ['number layers: {}'.format(nVertLevels)]

    if config.has_option('vertical_grid', 'bottom_depth'):
        bottom_depth = config.getfloat('vertical_grid', 'bottom_depth')
        txt.extend(
            ['bottom depth requested:  {:8.2f}'.format(bottom_depth),
             'bottom depth actual:     {:8.2f}'.format(np.amax(botDepth))])

    if config.has_option('vertical_grid', 'min_layer_thickness'):
        min_layer_thickness = config.getfloat('vertical_grid',
                                              'min_layer_thickness')
        txt.extend(
            ['min thickness requested: {:8.2f}'.format(min_layer_thickness),
             'min thickness actual:    {:8.2f}'.format(
                 np.amin(layerThickness[:]))])

    if config.has_option('vertical_grid', 'max_layer_thickness'):
        max_layer_thickness = config.getfloat('vertical_grid',
                                              'max_layer_thickness')
        txt.extend(
            ['max thickness requested: {:8.2f}'.format(max_layer_thickness),
             'max thickness actual:    {:8.2f}'.format(
                 np.amax(layerThickness[:]))])

    txt = '\n'.join(txt)
    print(txt)
    plt.subplot(2, 2, 4)
    plt.text(0, 0, txt, fontsize=12)
    plt.axis('off')
    plt.savefig(out_filename)
