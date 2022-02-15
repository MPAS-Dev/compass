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
                   dist_to_edge=None, dist_to_grounding_line=None):
    """
    Set cell widths based on settings in config file to pass to
    :py:func:`mpas_tools.mesh.creation.build_mesh.build_planar_mesh()`.
    Requires the following options to be set in the given config section:
    ``min_spac``, ``max_spac``, ``high_log_speed``, ``low_log_speed``,
    ``high_dist``, ``low_dist``,``cull_distance``, ``use_speed``,
    ``use_dist_to_edge``, and ``use_dist_to_grounding_line``.

    Parameters
    ----------
    section : str
        Section of the config file from which to read parameters
    thk : numpy.ndarray
        Ice thickness field from gridded dataset,
        usually after trimming to flood fill mask
    vx : numpy.ndarray, optional
        x-component of ice velocity from gridded dataset,
        usually after trimming to flood fill mask. Can be set to ``None``
        if ``use_speed == False`` in config file.
    vy : numpy.ndarray, optional
        y-component of ice velocity from gridded dataset,
        usually after trimming to flood fill mask. Can be set to ``None``
        if ``use_speed == False`` in config file.
    dist_to_edge : numpy.ndarray, optional
        Distance from each cell to ice edge, calculated in separate function.
        Can be set to ``None`` if ``use_dist_to_edge == False`` in config file
        and you do not want to set large ``cell_width`` where cells will be
        culled anyway, but this is not recommended.
    dist_to_grounding_line : numpy.ndarray, optional
        Distance from each cell to grounding line, calculated in separate
        function.  Can be set to ``None`` if
        ``use_dist_to_grounding_line == False`` in config file.

    Returns
    -------
    cell_width : numpy.ndarray
        Desired width of MPAS cells based on mesh desnity functions to pass to
        :py:func:`mpas_tools.mesh.creation.build_mesh.build_planar_mesh()`.
    """

    logger = self.logger
    section = self.config[section]

    # Get config inputs for cell spacing functions
    min_spac = float(section.get('min_spac'))
    max_spac = float(section.get('max_spac'))
    high_log_speed = float(section.get('high_log_speed'))
    low_log_speed = float(section.get('low_log_speed'))
    high_dist = float(section.get('high_dist'))
    low_dist = float(section.get('low_dist'))
    # convert km to m
    cull_distance = float(section.get('cull_distance')) * 1.e3

    # Make cell spacing function mapping from log speed to cell spacing
    if section.get('use_speed') == 'True':
        logger.info('Using speed for cell spacing')
        speed = (vx**2 + vy**2)**0.5
        lspd = np.log10(speed)
        spacing = np.interp(lspd, [low_log_speed, high_log_speed],
                            [max_spac, min_spac], left=max_spac,
                            right=min_spac)
        spacing[thk == 0.0] = min_spac
    else:
        spacing = max_spac*np.ones_like(thk)

    # Make cell spacing function mapping from distance to ice edge
    if section.get('use_dist_to_edge') == 'True':
        logger.info('Using distance to ice edge for cell spacing')
        spacing2 = np.interp(dist_to_edge, [low_dist, high_dist],
                             [min_spac, max_spac], left=min_spac,
                             right=max_spac)
        spacing2[thk == 0.0] = min_spac
    else:
        spacing2 = max_spac*np.ones_like(thk)

    # Make cell spacing function mapping from distance to grounding line
    if section.get('use_dist_to_grounding_line') == 'True':
        logger.info('Using distance to grounding line for cell spacing')
        spacing3 = np.interp(dist_to_grounding_line, [low_dist, high_dist],
                             [min_spac, max_spac], left=min_spac,
                             right=max_spac)
        spacing3[thk == 0.0] = min_spac
    else:
        spacing3 = max_spac*np.ones_like(thk)

    # Merge cell spacing methods
    cell_width = np.minimum(spacing, spacing2)
    cell_width = np.minimum(cell_width, spacing3)

    # Set large cell_width in areas we are going to cull anyway (speeds up
    # whole process). Use 10x the cull_distance to avoid this affecting
    # cell size in the final mesh. There may be a more rigorous way to set
    # that distance.
    if dist_to_edge is not None:
        cell_width[np.logical_and(thk == 0.0,
                   dist_to_edge > (10. * cull_distance))] = max_spac

    return cell_width


def get_dist_to_edge_and_GL(thk, topg, x, y, windowSize=1.e5):
    """
    Calculate distance from each point to ice edge and grounding line,
    to be used in mesh density functions in
    :py:func:`compass.landice.mesh.set_cell_width()`.

    Parameters
    ----------
    thk : numpy.ndarray
        Ice thickness field from gridded dataset,
        usually after trimming to flood fill mask
    topg : numpy.ndarray
        Bed topography field from gridded dataset
    x : numpy.ndarray
        x coordinates from gridded dataset
    y : numpy.ndarray
        y coordinates from gridded dataset

    Returns
    -------
    distToEdge : numpy.ndarray
        Distance from each cell to the ice edge
    distToGroundingLine : numpy.ndarray
        Distance from each cell to the grounding line
    """

    dx = x[1] - x[0]  # assumed constant and equal in x and y
    nx = len(x)
    ny = len(y)
    sz = thk.shape

    # Create masks to define ice edge and grounding line
    neighbors = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                          [1, 1], [-1, 1], [1, -1], [-1, -1]])

    iceMask = thk > 0.0
    groundedMask = thk > (-1028.0 / 910.0 * topg)
    floatingMask = np.logical_and(thk < (-1028.0 /
                                         910.0 * topg), thk > 0.0)
    marginMask = np.zeros(sz, dtype='i')
    groundingLineMask = np.zeros(sz, dtype='i')

    for n in neighbors:
        notIceMask = np.logical_not(np.roll(iceMask, n, axis=[0, 1]))
        marginMask = np.logical_or(marginMask, notIceMask)

        notGroundedMask = np.logical_not(np.roll(groundedMask,
                                                 n, axis=[0, 1]))
        groundingLineMask = np.logical_or(groundingLineMask,
                                          notGroundedMask)

    # where ice exists and neighbors non-ice locations
    marginMask = np.logical_and(marginMask, iceMask)
    # optional - plot mask
    # plt.pcolor(marginMask); plt.show()

    # Calculate dist to margin and grounding line
    [XPOS, YPOS] = np.meshgrid(x, y)
    distToEdge = np.zeros(sz)
    distToGroundingLine = np.zeros(sz)

    d = int(np.ceil(windowSize / dx))
    rng = np.arange(-1*d, d, dtype='i')
    maxdist = float(d) * dx

    # just look over areas with ice
    # ind = np.where(np.ravel(thk, order='F') > 0)[0]
    ind = np.where(np.ravel(thk, order='F') >= 0)[0]  # do it everywhere
    for iii in range(len(ind)):
        [i, j] = np.unravel_index(ind[iii], sz, order='F')

        irng = i + rng
        jrng = j + rng

        # only keep indices in the grid
        irng = irng[np.nonzero(np.logical_and(irng >= 0, irng < ny))]
        jrng = jrng[np.nonzero(np.logical_and(jrng >= 0, jrng < nx))]

        dist2Here = ((XPOS[np.ix_(irng, jrng)] - x[j])**2 +
                     (YPOS[np.ix_(irng, jrng)] - y[i])**2)**0.5

        dist2HereEdge = dist2Here.copy()
        dist2HereGroundingLine = dist2Here.copy()

        dist2HereEdge[marginMask[np.ix_(irng, jrng)] == 0] = maxdist
        dist2HereGroundingLine[groundingLineMask
                               [np.ix_(irng, jrng)] == 0] = maxdist

        distToEdge[i, j] = dist2HereEdge.min()
        distToGroundingLine[i, j] = dist2HereGroundingLine.min()

    return distToEdge, distToGroundingLine
