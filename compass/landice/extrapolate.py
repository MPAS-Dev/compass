import sys

import numpy as np
from netCDF4 import Dataset


def extrapolate_variable(nc_file, var_name, extrap_method, set_value=None):
    """
    Function to extrapolate variable values into undefined regions

    Parameters
    ----------
    nc_file : str
        the mpas file to modify

    var_name : str
        the mpas variable to extrapolate

    extrap_method : str
        idw, min, or value method of extrapolation

    set_value : float
        value to set variable to outside keepCellMask
        when using -v value
    """

    dataset = Dataset(nc_file, 'r+')
    varValue = dataset.variables[var_name][0, :]
    # Extrapolation
    nCells = len(dataset.dimensions['nCells'])
    if 'thickness' in dataset.variables.keys():
        thickness = dataset.variables['thickness'][0, :]
        bed = dataset.variables["bedTopography"][0, :]
    cellsOnCell = dataset.variables['cellsOnCell'][:]
    nEdgesOnCell = dataset.variables['nEdgesOnCell'][:]
    xCell = dataset.variables["yCell"][:]
    yCell = dataset.variables["xCell"][:]

    # Define region of good data to extrapolate from.
    # Different methods for different variables
    if var_name in ["effectivePressure", "beta", "muFriction"]:
        groundedMask = (thickness > (-1028.0 / 910.0 * bed))
        keepCellMask = np.copy(groundedMask)
        extrap_method == "min"

        # grow mask by one cell oceanward of GL
        for iCell in range(nCells):
            for n in range(nEdgesOnCell[iCell]):
                jCell = cellsOnCell[iCell, n] - 1
                if (groundedMask[jCell] == 1):
                    keepCellMask[iCell] = 1
                    continue
        # ensure zero muFriction does not get extrapolated
        keepCellMask *= (varValue > 0)
    elif var_name in ["floatingBasalMassBal"]:
        floatingMask = (thickness <= (-1028.0 / 910.0 * bed))
        keepCellMask = floatingMask * (varValue != 0.0)
        extrap_method == "idw"
    else:
        keepCellMask = (thickness > 0.0)

    # make a copy to edit that will be used later
    keepCellMaskNew = np.copy(keepCellMask)

    # recursive extrapolation steps:
    # 1) find cell A with mask = 0
    # 2) find how many surrounding cells have nonzero mask, and their
    #    indices (this will give us the cells from upstream)
    # 3) use the values for nonzero mask upstream cells to extrapolate
    #    the value for cell A
    # 4) change the mask for A from 0 to 1
    # 5) Update mask
    # 6) go to step 1)

    print("Start {} extrapolation using {} method".format(var_name,
                                                          extrap_method))
    if extrap_method == 'value':
        varValue[np.where(np.logical_not(keepCellMask))] = float(set_value)
    else:
        while np.count_nonzero(keepCellMask) != nCells:
            keepCellMask = np.copy(keepCellMaskNew)
            searchCells = np.where(keepCellMask == 0)[0]
            varValueOld = np.copy(varValue)

            for iCell in searchCells:
                neighborcellID = cellsOnCell[iCell, :nEdgesOnCell[iCell]] - 1
                # Important: ignore the phantom "neighbors" that are off
                # the edge of the mesh (0 values in cellsOnCell)
                neighborcellID = neighborcellID[neighborcellID >= 0]

                mask_for_idx = keepCellMask[neighborcellID]  # active cellmask
                mask_nonzero_idx, = np.nonzero(mask_for_idx)

                # id for nonzero beta cells
                nonzero_id = neighborcellID[mask_nonzero_idx]
                nonzero_num = np.count_nonzero(mask_for_idx)

                assert len(nonzero_id) == nonzero_num

                if nonzero_num > 0:
                    x_i = xCell[iCell]
                    y_i = yCell[iCell]
                    x_adj = xCell[nonzero_id]
                    y_adj = yCell[nonzero_id]
                    var_adj = varValueOld[nonzero_id]
                    if extrap_method == 'idw':
                        ds = np.sqrt((x_i - x_adj)**2 + (y_i - y_adj)**2)
                        assert np.count_nonzero(ds) == len(ds)
                        var_interp = 1.0 / sum(1.0 / ds) * \
                            sum(1.0 / ds * var_adj)
                        varValue[iCell] = var_interp
                    elif extrap_method == 'min':
                        varValue[iCell] = np.min(var_adj)
                    else:
                        sys.exit("ERROR: invalid extrapolation scheme! "
                                 "Set option m as idw or min!")

                    keepCellMaskNew[iCell] = 1

            # print ("{0:8d} cells left for extrapolation in total {1:8d} "
            # "cells".format(nCells-np.count_nonzero(keepCellMask),  nCells))

    # Put updated array back into file
    dataset.variables[var_name][0, :] = varValue
    dataset.close()
