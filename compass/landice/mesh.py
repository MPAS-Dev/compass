import numpy as np
import jigsawpy


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


def set_rectangular_geom_points_and_edges(xmin, xmax, ymin, ymax):
    """
    Set node and edge coordinates to pass to
    :py:func:`mpas_tools.mesh.creation.build_mesh.build_planar_mesh()`.

    Parameters
    ----------
    xmin : int or float
        Left-most x-coordinate in region to mesh
    xmax : int or float
        Right-most x-coordinate in region to mesh
    ymin : int or float
        Bottom-most y-coordinate in region to mesh
    ymax : int or float
        Top-most y-coordinate in region to mesh

    Returns
    -------
    geom_points : jigsawpy.jigsaw_msh_t.VERT2_t
        xy node coordinates to pass to build_planar_mesh()
    geom_edges : jigsawpy.jigsaw_msh_t.EDGE2_t
        xy edge coordinates between nodes to pass to build_planar_mesh()
    """

    geom_points = np.array([  # list of xy "node" coordinates
        ((xmin, ymin), 0),
        ((xmax, ymin), 0),
        ((xmax, ymax), 0),
        ((xmin, ymax), 0)],
        dtype=jigsawpy.jigsaw_msh_t.VERT2_t)

    geom_edges = np.array([  # list of "edges" between nodes
        ((0, 1), 0),
        ((1, 2), 0),
        ((2, 3), 0),
        ((3, 0), 0)],
        dtype=jigsawpy.jigsaw_msh_t.EDGE2_t)

    return geom_points, geom_edges


def set_cell_width(self, section, thk, vx=None, vy=None,
                   distToEdge=None, distToGroundingLine=None):
    """
    Set cell widths based on settings in config file to pass to
    :py:func:`mpas_tools.mesh.creation.build_mesh.build_planar_mesh()`.

    Parameters
    ----------
    section : str
        Section of the config file from which to read parameters
    thk : numpy.ndarray
        Ice thickness field from gridded dataset,
        usually after trimming to flood fill mask
    vx : numpy.ndarray
        x-component of ice velocity from gridded dataset,
        usually after trimming to flood fill mask. Can be set to None
        if useSpeed == 'False' in config file.
    vy : numpy.ndarray
        y-component of ice velocity from gridded dataset,
        usually after trimming to flood fill mask. Can be set to None
        if useSpeed == 'False' in config file.
    distToEdge : numpy.ndarray
        Distance from each cell to ice edge, calculated in separate function.
        Can be set to None if useDistToEdge == 'False' in config file and you
        do not want to set large cell_width where cells will be culled anyway,
        but this is not recommended.
    distToGroundingLine : numpy.ndarray
        Distance from each cell to grounding line, calculated in separate
        function.  Can be set to None if useDistToGroundingLine == 'False'
        in config file.

    Returns
    -------
    cell_width : numpy.ndarray
        Desired width of MPAS cells based on mesh desnity functions to pass to
        :py:func:`mpas_tools.mesh.creation.build_mesh.build_planar_mesh()`.
    """

    logger = self.logger
    section = self.config[section]

    # Get config inputs for cell spacing functions
    minSpac = float(section.get('minSpac'))
    maxSpac = float(section.get('maxSpac'))
    highLogSpeed = float(section.get('highLogSpeed'))
    lowLogSpeed = float(section.get('lowLogSpeed'))
    highDist = float(section.get('highDist'))
    lowDist = float(section.get('lowDist'))
    cullDistance = float(section.get('cullDistance')) * 1.e3  # convert km to m

    # Make cell spacing function mapping from log speed to cell spacing
    if section.get('useSpeed') == 'True':
        logger.info('Using speed for cell spacing')
        speed = (vx**2 + vy**2)**0.5
        lspd = np.log10(speed)
        spacing = np.interp(lspd, [lowLogSpeed, highLogSpeed],
                            [maxSpac, minSpac], left=maxSpac,
                            right=minSpac)
        spacing[thk == 0.0] = minSpac
    else:
        spacing = thk * 0. + maxSpac

    # Make cell spacing function mapping from distance to ice edge
    if section.get('useDistToEdge') == 'True':
        logger.info('Using distance to ice edge for cell spacing')
        spacing2 = np.interp(distToEdge, [lowDist, highDist],
                             [minSpac, maxSpac], left=minSpac,
                             right=maxSpac)
        spacing2[thk == 0.0] = minSpac
    else:
        spacing2 = thk * 0. + maxSpac

    # Make cell spacing function mapping from distance to grounding line
    if section.get('useDistToGroundingLine') == 'True':
        logger.info('Using distance to grounding line for cell spacing')
        spacing3 = np.interp(distToGroundingLine, [lowDist, highDist],
                             [minSpac, maxSpac], left=minSpac,
                             right=maxSpac)
        spacing3[thk == 0.0] = minSpac
    else:
        spacing3 = thk * 0. + maxSpac

    # Merge cell spacing methods
    cell_width = np.minimum(spacing, spacing2)
    cell_width = np.minimum(cell_width, spacing3)

    # Set large cell_width in areas we are going to cull anyway (speeds up
    # whole process). Use 10x the cullDistance to avoid this affecting
    # cell size in the final mesh. There may be a more rigorous way to set
    # that distance.
    if distToEdge is not None:
        cell_width[np.logical_and(thk == 0.0,
                   distToEdge > (10. * cullDistance))] = maxSpac

    return cell_width
