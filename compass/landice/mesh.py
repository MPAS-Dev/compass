import numpy as np


def gridded_flood_fill(field):
    """
    Generic flood-fill routine to create mask of connected elements
    in the desired input array (field) from a gridded dataset. This
    is generally used to remove glaciers and ice-fields that are not
    connected to the ice sheet. Note that there may be more efficient
    algorithms.

    Parameters
    ----------
    field : numpy.ndarray
        Array from gridded dataset to use for flood-fill.
        Usually ice thickness.

    Returns
    -------
    floodMask : numpy.ndarray
        Mask calculated by the flood fill routine,
        where cells connected to the ice sheet (or main feature)
        are 1 and everything else is 0.
    """

    sz = field.shape
    searchedMask = np.zeros(sz)
    floodMask = np.zeros(sz)
    iStart = sz[0] // 2
    jStart = sz[1] // 2
    floodMask[iStart, jStart] = 1

    neighbors = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    lastSearchList = np.ravel_multi_index([[iStart], [jStart]],
                                          sz, order='F')

    cnt = 0
    while len(lastSearchList) > 0:
        cnt += 1
        newSearchList = np.array([], dtype='i')

        for iii in range(len(lastSearchList)):
            [i, j] = np.unravel_index(lastSearchList[iii], sz, order='F')
            # search neighbors
            for n in neighbors:
                ii = i + n[0]
                jj = j + n[1]  # subscripts to neighbor
                # only consider unsearched neighbors
                if searchedMask[ii, jj] == 0:
                    searchedMask[ii, jj] = 1  # mark as searched

                    if field[ii, jj] > 0.0:
                        floodMask[ii, jj] = 1  # mark as ice
                        # add to list of newly found  cells
                        newSearchList = np.append(newSearchList,
                                                  np.ravel_multi_index(
                                                      [[ii], [jj]], sz,
                                                      order='F')[0])
        lastSearchList = newSearchList

    return floodMask
