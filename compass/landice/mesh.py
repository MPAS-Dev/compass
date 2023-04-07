import time

import jigsawpy
import numpy as np
import xarray
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.mesh.creation import build_planar_mesh
from netCDF4 import Dataset


def gridded_flood_fill(field, iStart=None, jStart=None):
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
    iStart : int
        x index from which to start flood fill for field.
        Defaults to the center x coordinate.
    jStart : int
        y index from which to start flood fill.
        Defaults to the center y coordinate.

    Returns
    -------
    flood_mask : numpy.ndarray
        _mask calculated by the flood fill routine,
        where cells connected to the ice sheet (or main feature)
        are 1 and everything else is 0.
    """

    sz = field.shape
    searched_mask = np.zeros(sz)
    flood_mask = np.zeros(sz)
    if iStart is None and jStart is None:
        iStart = sz[0] // 2
        jStart = sz[1] // 2
    flood_mask[iStart, jStart] = 1

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
                ii = min(i + n[0], sz[0] - 1)  # don't go out of bounds
                jj = min(j + n[1], sz[1] - 1)  # subscripts to neighbor
                # only consider unsearched neighbors
                if searched_mask[ii, jj] == 0:
                    searched_mask[ii, jj] = 1  # mark as searched

                    if field[ii, jj] > 0.0:
                        flood_mask[ii, jj] = 1  # mark as ice
                        # add to list of newly found  cells
                        newSearchList = np.append(newSearchList,
                                                  np.ravel_multi_index(
                                                      [[ii], [jj]], sz,
                                                      mode='clip',
                                                      order='F')[0])
        lastSearchList = newSearchList

    return flood_mask


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


def set_cell_width(self, section, thk, bed=None, vx=None, vy=None,
                   dist_to_edge=None, dist_to_grounding_line=None,
                   flood_fill_iStart=None, flood_fill_jStart=None):
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
    bed : numpy.ndarray
        Bed topography from gridded dataset
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
    flood_fill_iStart : int, optional
        x-index location to start flood-fill when using bed topography
    flood_fill_jStart : int, optional
        y-index location to start flood-fill when using bed topography

    Returns
    -------
    cell_width : numpy.ndarray
        Desired width of MPAS cells based on mesh desnity functions to pass to
        :py:func:`mpas_tools.mesh.creation.build_mesh.build_planar_mesh()`.
    """

    logger = self.logger
    section = self.config[section]

    # Get config inputs for cell spacing functions
    min_spac = section.getfloat('min_spac')
    max_spac = section.getfloat('max_spac')
    high_log_speed = section.getfloat('high_log_speed')
    low_log_speed = section.getfloat('low_log_speed')
    high_dist = section.getfloat('high_dist')
    low_dist = section.getfloat('low_dist')
    high_dist_bed = section.getfloat('high_dist_bed')
    low_dist_bed = section.getfloat('low_dist_bed')
    low_bed = section.getfloat('low_bed')
    high_bed = section.getfloat('high_bed')

    # convert km to m
    cull_distance = float(section.get('cull_distance')) * 1.e3

    # Cell spacing function based on union of masks
    if section.get('use_bed') == 'True':
        logger.info('Using bed elevation for spacing.')
        if flood_fill_iStart is not None and flood_fill_jStart is not None:
            logger.info('calling gridded_flood_fill to find \
                        bedTopography <= low_bed connected to the ocean.')
            tic = time.time()
            # initialize mask to low bed topography
            in_mask = (bed <= low_bed)
            # Do not let flood fill reach further than high_dist_bed into
            # the ice sheet interior.
            in_mask[np.logical_and(
                thk > 0, dist_to_grounding_line >= high_dist_bed)] = 0
            low_bed_mask = gridded_flood_fill(in_mask,
                                              iStart=flood_fill_iStart,
                                              jStart=flood_fill_jStart)
            toc = time.time()
            logger.info(f'Flood fill finished in {toc - tic} seconds.')
        # Use a logistics curve for bed topography spacing.
        k = 0.05  # This works well, but could try other values
        spacing_bed = min_spac + (max_spac - min_spac) / (1.0 + np.exp(
            -k * (bed - np.mean([high_bed, low_bed]))))
        # We only want bed topography to influence spacing within high_dist_bed
        # from the ice margin. In the region between high_dist_bed and
        # low_dist_bed, use a linear ramp to damp influence of bed topo.
        spacing_bed[dist_to_grounding_line >= low_dist_bed] = (
            (1.0 - (dist_to_grounding_line[
                dist_to_grounding_line >= low_dist_bed] -
                low_dist_bed) / (high_dist_bed - low_dist_bed)) *
            spacing_bed[dist_to_grounding_line >= low_dist_bed] +
            (dist_to_grounding_line[dist_to_grounding_line >=
                                    low_dist_bed] - low_dist_bed) /
            (high_dist_bed - low_dist_bed) * max_spac)
        spacing_bed[dist_to_grounding_line >= high_dist_bed] = max_spac
        if flood_fill_iStart is not None and flood_fill_jStart is not None:
            spacing_bed[low_bed_mask == 0] = max_spac
            # Do one more flood fill to eliminate isolated pockets
            # of high resolution that were separated when we set
            # spacing_bed[dist_to_grounding_line >= high_dist_bed] = max_spac
            in_mask2 = (bed <= low_bed)
            in_mask2[np.logical_and(
                thk > 0, spacing_bed > (2. * min_spac))] = 0
            low_bed_mask2 = gridded_flood_fill(in_mask2,
                                               iStart=flood_fill_iStart,
                                               jStart=flood_fill_jStart)
            spacing_bed[low_bed_mask2 == 0] = max_spac
    else:
        spacing_bed = max_spac * np.ones_like(thk)

    # Make cell spacing function mapping from log speed to cell spacing
    if section.get('use_speed') == 'True':
        logger.info('Using speed for cell spacing')
        speed = (vx ** 2 + vy ** 2) ** 0.5
        lspd = np.log10(speed)
        spacing_speed = np.interp(lspd, [low_log_speed, high_log_speed],
                                  [max_spac, min_spac], left=max_spac,
                                  right=min_spac)

        # Clean up where we have missing velocities. These are usually nans
        # or the default netCDF _FillValue of ~10.e36
        missing_data_mask = np.logical_or(
            np.logical_or(np.isnan(vx), np.isnan(vy)),
            np.logical_or(np.abs(vx) > 1.e5,
                          np.abs(vy) > 1.e5))
        spacing_speed[missing_data_mask] = max_spac
        logger.info(f'Found {np.sum(missing_data_mask)} points in input '
                    f'dataset with missing velocity values. Setting '
                    f'velocity-based spacing to maximum value.')

        spacing_speed[thk == 0.0] = min_spac
    else:
        spacing_speed = max_spac * np.ones_like(thk)

    # Make cell spacing function mapping from distance to ice edge
    if section.get('use_dist_to_edge') == 'True':
        logger.info('Using distance to ice edge for cell spacing')
        spacing_edge = np.interp(dist_to_edge, [low_dist, high_dist],
                                 [min_spac, max_spac], left=min_spac,
                                 right=max_spac)
        spacing_edge[thk == 0.0] = min_spac
    else:
        spacing_edge = max_spac * np.ones_like(thk)

    # Make cell spacing function mapping from distance to grounding line
    if section.get('use_dist_to_grounding_line') == 'True':
        logger.info('Using distance to grounding line for cell spacing')
        spacing_gl = np.interp(dist_to_grounding_line, [low_dist, high_dist],
                               [min_spac, max_spac], left=min_spac,
                               right=max_spac)
        spacing_gl[thk == 0.0] = min_spac
    else:
        spacing_gl = max_spac * np.ones_like(thk)

    # Merge cell spacing methods
    cell_width = max_spac * np.ones_like(thk)
    for width in [spacing_bed, spacing_speed, spacing_edge, spacing_gl]:
        cell_width = np.minimum(cell_width, width)

    # Set large cell_width in areas we are going to cull anyway (speeds up
    # whole process). Use 3x the cull_distance to avoid this affecting
    # cell size in the final mesh. There may be a more rigorous way to set
    # that distance.
    if dist_to_edge is not None:
        mask = np.logical_and(
            thk == 0.0, dist_to_edge > (3. * cull_distance))
        cell_width[mask] = max_spac

    return cell_width


def get_dist_to_edge_and_GL(self, thk, topg, x, y, section, window_size=None):
    """
    Calculate distance from each point to ice edge and grounding line,
    to be used in mesh density functions in
    :py:func:`compass.landice.mesh.set_cell_width()`. In future development,
    this should be updated to use a faster package such as `scikit-fmm`.

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
    section : str
        section of config file used to define mesh parameters
    window_size : int or float
        Size (in meters) of a search 'box' (one-directional) to use
        to calculate the distance from each cell to the ice margin.
        Bigger number makes search slower, but if too small, the transition
        zone could get truncated. We usually want this calculated as the
        maximum of high_dist and high_dist_bed, but there may be cases in
        which it is useful to set it manually. However, it should never be
        smaller than either high_dist or high_dist_bed.

    Returns
    -------
    dist_to_edge : numpy.ndarray
        Distance from each cell to the ice edge
    dist_to_grounding_line : numpy.ndarray
        Distance from each cell to the grounding line
    """
    logger = self.logger
    section = self.config[section]
    tic = time.time()

    high_dist = float(section.get('high_dist'))
    high_dist_bed = float(section.get('high_dist_bed'))

    if window_size is None:
        window_size = max(high_dist, high_dist_bed)
    elif window_size < min(high_dist, high_dist_bed):
        logger.info('WARNING: window_size was set to a value smaller'
                    ' than high_dist and/or high_dist_bed. Resetting'
                    f' window_size to {max(high_dist, high_dist_bed)},'
                    ' which is  max(high_dist, high_dist_bed)')
        window_size = max(high_dist, high_dist_bed)

    dx = x[1] - x[0]  # assumed constant and equal in x and y
    nx = len(x)
    ny = len(y)
    sz = thk.shape

    # Create masks to define ice edge and grounding line
    neighbors = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                          [1, 1], [-1, 1], [1, -1], [-1, -1]])

    ice_mask = thk > 0.0
    grounded_mask = thk > (-1028.0 / 910.0 * topg)
    margin_mask = np.zeros(sz, dtype='i')
    grounding_line_mask = np.zeros(sz, dtype='i')

    for n in neighbors:
        not_ice_mask = np.logical_not(np.roll(ice_mask, n, axis=[0, 1]))
        margin_mask = np.logical_or(margin_mask, not_ice_mask)

        not_grounded_mask = np.logical_not(np.roll(grounded_mask,
                                                   n, axis=[0, 1]))
        grounding_line_mask = np.logical_or(grounding_line_mask,
                                            not_grounded_mask)

    # where ice exists and neighbors non-ice locations
    margin_mask = np.logical_and(margin_mask, ice_mask)
    # optional - plot mask
    # plt.pcolor(margin_mask); plt.show()

    # Calculate dist to margin and grounding line
    [XPOS, YPOS] = np.meshgrid(x, y)
    dist_to_edge = np.zeros(sz)
    dist_to_grounding_line = np.zeros(sz)

    d = int(np.ceil(window_size / dx))
    rng = np.arange(-1 * d, d, dtype='i')
    max_dist = float(d) * dx

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

        dist_to_here = ((XPOS[np.ix_(irng, jrng)] - x[j]) ** 2 +
                        (YPOS[np.ix_(irng, jrng)] - y[i]) ** 2) ** 0.5

        dist_to_here_edge = dist_to_here.copy()
        dist_to_here_grounding_line = dist_to_here.copy()

        dist_to_here_edge[margin_mask[np.ix_(irng, jrng)] == 0] = max_dist
        dist_to_here_grounding_line[grounding_line_mask
                                    [np.ix_(irng, jrng)] == 0] = max_dist

        dist_to_edge[i, j] = dist_to_here_edge.min()
        dist_to_grounding_line[i, j] = dist_to_here_grounding_line.min()

    toc = time.time()
    logger.info('compass.landice.mesh.get_dist_to_edge_and_GL() took {:0.2f} '
                'seconds'.format(toc - tic))

    return dist_to_edge, dist_to_grounding_line


def build_cell_width(self, section_name, gridded_dataset,
                     flood_fill_start=[None, None]):
    """
    Determine MPAS mesh cell size based on user-defined density function.

    Parameters
    ----------
    section : str
        section of config file used to define mesh parameters
    gridded_dataset : str
        name of .nc file used to define cell spacing
    flood_fill_start : list of ints
        i and j indices used to define starting location for flood fill.
        Most cases will use [None, None], which will just start the flood
        fill in the center of the gridded dataset.

    Returns
    -------
    cell_width : numpy.ndarray
        Desired width of MPAS cells based on mesh desnity functions to pass to
        :py:func:`mpas_tools.mesh.creation.build_mesh.build_planar_mesh()`.
    x1 : float
        x coordinates from gridded dataset
    y1 : float
        y coordinates from gridded dataset
    geom_points : jigsawpy.jigsaw_msh_t.VERT2_t
        xy node coordinates to pass to build_planar_mesh()
    geom_edges : jigsawpy.jigsaw_msh_t.EDGE2_t
        xy edge coordinates between nodes to pass to build_planar_mesh()
    flood_mask : numpy.ndarray
        mask calculated by the flood fill routine,
        where cells connected to the ice sheet (or main feature)
        are 1 and everything else is 0.
    """

    section = self.config[section_name]
    # get needed fields from gridded dataset
    f = Dataset(gridded_dataset, 'r')
    f.set_auto_mask(False)  # disable masked arrays

    x1 = f.variables['x1'][:]
    y1 = f.variables['y1'][:]
    thk = f.variables['thk'][0, :, :]
    topg = f.variables['topg'][0, :, :]
    vx = f.variables['vx'][0, :, :]
    vy = f.variables['vy'][0, :, :]

    f.close()

    # Define extent of region to mesh.
    xx0 = section.get('x_min')
    xx1 = section.get('x_max')
    yy0 = section.get('y_min')
    yy1 = section.get('y_max')

    if 'None' in [xx0, xx1, yy0, yy1]:
        xx0 = np.min(x1)
        xx1 = np.max(x1)
        yy0 = np.min(y1)
        yy1 = np.max(y1)
    else:
        xx0 = float(xx0)
        xx1 = float(xx1)
        yy0 = float(yy0)
        yy1 = float(yy1)

    geom_points, geom_edges = set_rectangular_geom_points_and_edges(
        xx0, xx1, yy0, yy1)

    # Remove ice not connected to the ice sheet.
    flood_mask = gridded_flood_fill(thk)
    thk[flood_mask == 0] = 0.0
    vx[flood_mask == 0] = 0.0
    vy[flood_mask == 0] = 0.0

    # Calculate distance from each grid point to ice edge
    # and grounding line, for use in cell spacing functions.
    distToEdge, distToGL = get_dist_to_edge_and_GL(
        self, thk, topg, x1,
        y1, section=section_name)

    # Set cell widths based on mesh parameters set in config file
    cell_width = set_cell_width(self, section=section_name,
                                thk=thk, bed=topg, vx=vx, vy=vy,
                                dist_to_edge=distToEdge,
                                dist_to_grounding_line=distToGL,
                                flood_fill_iStart=flood_fill_start[0],
                                flood_fill_jStart=flood_fill_start[1])

    return (cell_width.astype('float64'), x1.astype('float64'),
            y1.astype('float64'), geom_points, geom_edges, flood_mask)


def build_MALI_mesh(self, cell_width, x1, y1, geom_points,
                    geom_edges, mesh_name, section_name,
                    gridded_dataset, projection, geojson_file=None):

    logger = self.logger
    section = self.config[section_name]

    logger.info('calling build_planar_mesh')
    build_planar_mesh(cell_width, x1, y1, geom_points,
                      geom_edges, logger=logger)
    dsMesh = xarray.open_dataset('base_mesh.nc')
    logger.info('culling mesh')
    dsMesh = cull(dsMesh, logger=logger)
    logger.info('converting to MPAS mesh')
    dsMesh = convert(dsMesh, logger=logger)
    logger.info('writing grid_converted.nc')
    write_netcdf(dsMesh, 'grid_converted.nc')
    levels = section.get('levels')
    logger.info('calling create_landice_grid_from_generic_MPAS_grid.py')
    args = ['create_landice_grid_from_generic_MPAS_grid.py',
            '-i', 'grid_converted.nc',
            '-o', 'grid_preCull.nc',
            '-l', levels, '-v', 'glimmer']
    check_call(args, logger=logger)

    logger.info('calling interpolate_to_mpasli_grid.py')
    args = ['interpolate_to_mpasli_grid.py', '-s',
            gridded_dataset, '-d',
            'grid_preCull.nc', '-m', 'b', '-t']

    check_call(args, logger=logger)

    cullDistance = section.get('cull_distance')
    if float(cullDistance) > 0.:
        logger.info('calling define_cullMask.py')
        args = ['define_cullMask.py', '-f',
                'grid_preCull.nc', '-m',
                'distance', '-d', cullDistance]

        check_call(args, logger=logger)
    else:
        logger.info('cullDistance <= 0 in config file. '
                    'Will not cull by distance to margin. \n')

    if geojson_file is not None:
        # This step is only necessary because the GeoJSON region
        # is defined by lat-lon.
        logger.info('calling set_lat_lon_fields_in_planar_grid.py')
        args = ['set_lat_lon_fields_in_planar_grid.py', '-f',
                'grid_preCull.nc', '-p', projection]

        check_call(args, logger=logger)

        logger.info('calling MpasMaskCreator.x')
        args = ['MpasMaskCreator.x', 'grid_preCull.nc',
                'mask.nc', '-f', geojson_file]

        check_call(args, logger=logger)

        logger.info('culling to geojson file')

    dsMesh = xarray.open_dataset('grid_preCull.nc')
    if geojson_file is not None:
        mask = xarray.open_dataset('mask.nc')
    else:
        mask = None

    dsMesh = cull(dsMesh, dsInverse=mask, logger=logger)
    write_netcdf(dsMesh, 'culled.nc')

    logger.info('Marking horns for culling')
    args = ['mark_horns_for_culling.py', '-f', 'culled.nc']
    check_call(args, logger=logger)

    logger.info('culling and converting')
    dsMesh = xarray.open_dataset('culled.nc')
    dsMesh = cull(dsMesh, logger=logger)
    dsMesh = convert(dsMesh, logger=logger)
    write_netcdf(dsMesh, 'dehorned.nc')

    logger.info('calling create_landice_grid_from_generic_MPAS_grid.py')
    args = ['create_landice_grid_from_generic_MPAS_grid.py', '-i',
            'dehorned.nc', '-o',
            mesh_name, '-l', levels, '-v', 'glimmer',
            '--beta', '--thermal', '--obs', '--diri']

    check_call(args, logger=logger)

    logger.info('calling interpolate_to_mpasli_grid.py')
    args = ['interpolate_to_mpasli_grid.py', '-s',
            gridded_dataset, '-d', mesh_name, '-m', 'b']
    check_call(args, logger=logger)

    logger.info('Marking domain boundaries dirichlet')
    args = ['mark_domain_boundaries_dirichlet.py',
            '-f', mesh_name]
    check_call(args, logger=logger)

    logger.info('calling set_lat_lon_fields_in_planar_grid.py')
    args = ['set_lat_lon_fields_in_planar_grid.py', '-f',
            mesh_name, '-p', projection]
    check_call(args, logger=logger)
