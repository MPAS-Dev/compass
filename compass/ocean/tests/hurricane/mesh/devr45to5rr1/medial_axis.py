
#####################################################################
#####################################################################

"""
Extract the longest medial axis (centerline) for each labeled island from a
NetCDF topography field. For each island (connected component in the binary
mask), the medial axis is computed and its skeleton components are identified.
Then, for each skeleton component a graph is built (in 8-connectivity) and the
geodesic “diameter” (longest path) is computed via a double-BFS.  Finally, for
the island the longest center line is selected and converted into a shapely
LineString.

Usage: python extract_longest_components.py <netcdf_file> [--topovar TOPOVAR]
[--threshold THRESHOLD] [--xvar XVAR] [--yvar YVAR]

If coordinate variables (xvar,yvar) are provided, the pixel indices will be
mapped to those coordinates; otherwise, pixel (row,col) indices (with x=col)
are used.
"""

import json
from collections import deque
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from shapely import vectorized
from shapely.geometry import LineString, Point, shape
from skimage.measure import label
from skimage.morphology import medial_axis, remove_small_objects

R = 6371220.0


#####################
# I/O and Preprocessing
#####################
def read_topography(nc_file, topo_var='topo', x_var=None, y_var=None):
    """
    Reads topography data from a NetCDF file.

    Arguments: nc_file: str, path to the NetCDF file.  topo_var: str, variable
    name for topography data.  x_var, y_var: Optional coordinate variable names

    Returns: topo: 2D numpy array of topography data.  X, Y: 2D coordinate
    arrays (or None) if coordinate variables are provided.
    """

    # Great South Bay
    # lonmin = -73.475
    # lonmax = -73.0
    # latmin = 40.60
    # latmax = 40.675

    # NY/NJ Bight
    lonmin = -74.2
    lonmax = -72.9
    latmin = 39.9
    latmax = 40.8

    ds_bathy = xr.open_dataset(nc_file)
    lon = ds_bathy.lon.values[:]
    lat = ds_bathy.lat.values[:]
    lon_idx, = np.where((lon >= lonmin) & (lon <= lonmax))
    lat_idx, = np.where((lat >= latmin) & (lat <= latmax))

    latlon_idx = np.ix_(lat_idx, lon_idx)
    topo = -ds_bathy.z.values[latlon_idx]
    print(topo.shape)

    x = np.linspace(lonmin, lonmax, topo.shape[1])
    y = np.linspace(latmin, latmax, topo.shape[0])
    X, Y = np.meshgrid(x, y)

    return topo, X, Y


def threshold_island(topo, threshold, regions):
    """
    Threshold the topography to create an island mask.  (Adjust inequality as
    needed; here islands are assumed to be above the threshold.)
    """

    island = topo < threshold
    # Optional: remove small objects if noise exists.
    island = remove_small_objects(island, min_size=200)
    island = np.logical_and(island, regions)

    return island


def compute_region_masks(geojson_file, lon_grid, lat_grid):
    """
    Computes a mask (boolean NumPy array) for each region (polygon) in the
    GeoJSON file.  The mask array has the same shape as the NetCDF grid.
    A value of True indicates that the point is within the region.

    Returns:
        A dictionary mapping region identifier to its computed mask.
        The region identifier is taken from the 'id' property if available,
        otherwise the index of the polygon.
    """

    # Load the GeoJSON file using geopandas
    with open(geojson_file, 'r') as f:
        geojson_data = json.load(f)
    features = geojson_data['features']

    region_masks = []
    region_mask = np.zeros_like(lon_grid)
    for idx, feature in enumerate(features):
        geom = shape(feature['geometry'])

        # Use shapely.vectorized.contains to compute a boolean mask.
        # Note: shapely.vectorized.contains expects x (lon) and y (lat) arrays.
        mask = vectorized.contains(geom, lon_grid, lat_grid)
        region_mask = np.logical_or(region_mask, mask)

        region_masks.append(mask)

    return region_mask


#####################
# Skeleton and Graph Functions
#####################
def compute_medial_axis_skel(mask):
    """
    Compute the medial axis (skeleton) of the binary mask.
    Returns a binary image.
    """
    skel = medial_axis(mask)
    return skel


def get_neighbors(coord, shape):
    """
    For a given pixel coordinate (r, c) yield all 8-connected neighbors
    (that lie inside the image).
    """
    r, c = coord
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                yield (nr, nc)


def build_skeleton_graph(skel_component):
    """
    Build an undirected graph from a binary skeleton component.
    Nodes are pixel coordinates where the mask is True and edges
    are added between 8-connected pixels.

    Returns a dict mapping node -> list of neighbor nodes.
    """
    graph = {}
    coords = np.transpose(np.nonzero(skel_component))
    pixel_set = set(tuple(coord) for coord in coords)
    shape = skel_component.shape
    for pix in pixel_set:
        neighbors = []
        for n in get_neighbors(pix, shape):
            if n in pixel_set:
                neighbors.append(n)
        graph[pix] = neighbors
    return graph


def bfs_furthest(graph, start):
    """
    Run BFS from the start node in the unweighted graph and return:
      - furthest node found (with the maximum distance)
      - distances dictionary
      - parent pointers.
    """
    queue = deque([start])
    distances = {start: 0}
    parent = {start: None}
    while queue:
        node = queue.popleft()
        for nbr in graph[node]:
            if nbr not in distances:
                distances[nbr] = distances[node] + 1
                parent[nbr] = node
                queue.append(nbr)
    furthest = max(distances, key=distances.get)
    return furthest, distances, parent


def retrieve_path(parent, start, goal):
    """
    Retrieve the (BFS) path from start to goal given the parent dictionary.
    """
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def compute_longest_geodesic_path(graph):
    """
    Compute the “diameter” (longest shortest path) of the given skeleton graph.
    We choose an arbitrary start (preferably an endpoint if available), perform
    BFS to get the furthest node A. Then, starting from A, perform a second BFS
    to get the furthest node B.  The path from A to B is the longest.  Returns
    the ordered list of nodes (pixels) along this path.
    """
    endpoints = [node for node, nbrs in graph.items() if len(nbrs) == 1]
    if endpoints:
        start = endpoints[0]
    else:
        # fallback if no endpoints exist (e.g., a closed loop)
        start = next(iter(graph))
    furthest_A, _, parent_A = bfs_furthest(graph, start)
    furthest_B, _, parent_B = bfs_furthest(graph, furthest_A)
    path = retrieve_path(parent_B, furthest_A, furthest_B)
    return path


def compute_geodesic_length(path):
    """
    Given an ordered list of pixel coordinates, compute the geodesic length.
    Here we use Euclidean distance between successive pixels.
    """
    if len(path) < 2:
        return 0.0
    length = 0.0
    for i in range(1, len(path)):
        (r1, c1), (r2, c2) = path[i - 1], path[i]
        length += sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2)
    return length


def centerline_to_linestring(path, X=None, Y=None):
    """
    Convert an ordered list of pixels into a shapely LineString.  If coordinate
    arrays (X, Y) are provided (2D arrays with same shape as the topo),
    map (row, col) to (X[row, col], Y[row, col]);
    otherwise, use (col, row) (x=col).
    """
    if X is not None and Y is not None:
        points = [(X[row, col], Y[row, col]) for row, col in path]
    else:
        points = [(col, row) for row, col in path]
    return LineString(points)


#####################
# Extraction for Each Label
#####################
def extract_longest_centerline_per_label(nc_file, geojson_file,
                                         topo_var='topo', threshold=0.0,
                                         x_var=None, y_var=None):
    """
    Process the NetCDF file to:
      - Read topography and threshold to create a binary island mask.
      - Label each island (connected component) of the island mask.
      - For each island label, compute its medial axis.
        Then, if a skeleton splits into multiple
        components, select the one with the longest geodesic (diameter) length.
      - Return a dict mapping the island label (integer) to its
        longest center line as a shapely LineString.
    """
    topo, X, Y = read_topography(nc_file, topo_var, x_var, y_var)
    regions = compute_region_masks(geojson_file, X, Y)
    island_mask = threshold_island(topo, threshold, regions)

    # Label islands (using connectivity=2 for 8-connectivity)
    island_labels = label(island_mask, connectivity=2)

    centerlines = []
    unique_labels = np.unique(island_labels)
    for label_val in unique_labels:
        if label_val == 0:
            continue
        # Isolate island component
        island_comp = (island_labels == label_val)
        # Compute skeleton for the island component
        skel = compute_medial_axis_skel(island_comp)
        # Label connected skeleton components (in case the skeleton fragments)
        skel_labels = label(skel, connectivity=2)
        longest_length = 0.0
        best_line = None
        for skel_lab in np.unique(skel_labels):
            if skel_lab == 0:
                continue
            comp_skel = (skel_labels == skel_lab)
            graph = build_skeleton_graph(comp_skel)
            if not graph:
                continue
            path = compute_longest_geodesic_path(graph)
            path_length = compute_geodesic_length(path)
            if path_length > longest_length:
                longest_length = path_length
                best_line = centerline_to_linestring(path, X, Y)
        if best_line is not None:
            n = len(list(best_line.xy[0]))
            if n > 200:
                centerlines.append(best_line)
    return centerlines, island_mask, X, Y


def sample_line_offsets_variable(centerline, spacing, offset_distance):
    """
    Sample points along a shapely LineString representing the centerline,
    using variable spacing.
    For each sample point, compute offset points to the left and right
    of the centerline.

    The 'spacing' parameter can be either:
      - a constant (number): in which case equal spacing is used, or
      - a callable function: spacing(d) returns the distance to add
        from the current cumulative
        distance along the centerline.
    The 'offset_distance' is the fixed perpendicular distance at which
    points will be placed on either side (left and right) of the centerline.
    The left side is defined by a 90°_counterclockwise rotation of the
    tangent vector.

    Parameters:
      centerline     : shapely LineString object.
      spacing        : float or callable; if callable,
                       spacing(current_distance) -> next step size.
      offset_distance: float, distance to offset the sample points
                       perpendicularly.

    Returns:
      (left_points, right_points) where each is a list of shapely Points.
    """
    left_points = []
    right_points = []

    total_length = centerline.length

    # Use a small value to compute the numerical derivative for the tangent.
    # epsilon = total_length * 1e-6 if total_length * 1e-6 > 1e-8 else 1e-6
    epsilon = spacing

    sample_dists = []
    d = 0.0
    while d < total_length:
        sample_dists.append(d)
        # Determine next spacing step
        if callable(spacing):
            step = spacing(d)
        else:
            step = spacing
        d += step
    # Ensure the last point (end of centerline) is included
    if sample_dists[-1] < total_length:
        sample_dists.append(total_length)

    for d in sample_dists:
        pt = centerline.interpolate(d)
        x, y = pt.x, pt.y

        # Estimate the tangent vector using a small forward
        # (or backward) offset.
        if d + epsilon <= total_length:
            pt_next = centerline.interpolate(d + epsilon)
            dx = pt_next.x - x
            dy = pt_next.y - y
        else:
            pt_prev = centerline.interpolate(max(d - epsilon, 0))
            dx = x - pt_prev.x
            dy = y - pt_prev.y

        norm = np.hypot(dx, dy)
        if norm == 0:
            continue
        tangent = (dx / norm, dy / norm)
        # Left normal is 90° counterclockwise rotation of tangent.
        normal = (-tangent[1], tangent[0])
        left_pt = Point(x + offset_distance * normal[0],
                        y + offset_distance * normal[1])
        right_pt = Point(x - offset_distance * normal[0],
                         y - offset_distance * normal[1])
        left_points.append(left_pt)
        right_points.append(right_pt)

    return left_points, right_points


def generate_offset_points_variable(centerlines, spacing, offset_distance):
    """
    Given an iterable of centerlines (shapely LineString objects),
    compute per-centerline offset points using variable spacing.

    Parameters:
      centerlines    : iterable of shapely LineString objects.
      spacing        : float or callable (as described in
                       sample_line_offsets_variable).
      offset_distance: float, the perpendicular offset distance.

    Returns:
      A dictionary mapping each centerline index to a tuple
      (left_points, right_points),  where left_points and right_points
      are lists of shapely Points.
    """
    all_points = []
    result = []
    for idx, cl in enumerate(centerlines):
        left_pts, right_pts = sample_line_offsets_variable(cl, spacing,
                                                           offset_distance)
        result.append((left_pts, right_pts))
        for pt in left_pts:
            all_points.append([pt.x, pt.y])
        for pt in right_pts:
            all_points.append([pt.x, pt.y])
    all_points = np.array(all_points)
    return all_points


def lonlat2xyz(lon, lat, R):

    x = R * np.cos(np.radians(lon)) * np.cos(np.radians(lat))
    y = R * np.sin(np.radians(lon)) * np.cos(np.radians(lat))
    z = R * np.sin(np.radians(lat))

    return x, y, z


def xyz2lonlat(x, y, z, R):

    lon = np.degrees(np.arcsin(z / R))
    lat = np.degress(np.arctan2(y / x))

    return lon, lat


def plot_linestrings(linestrings, image, X, Y, points):
    plt.figure(figsize=(10, 10))
    c = plt.contourf(X, Y, image, alpha=0.5)
    plt.colorbar(c)
    # plt.imshow(medial_axis, alpha=0.5)
    for line in linestrings:
        x, y = line.xy
        plt.plot(x, y, linewidth=2)
    plt.scatter(points[:, 0], points[:, 1], marker='.', color='k')
    plt.title('Medial Axis of Barrier Islands')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')
    plt.savefig('medial_axis.png')


#####################
# Command-line Interface
#####################
def main():
    nc_file = '/pscratch/sd/s/sbrus/compass_subgrid_hurricane_tides_isc/' \
              'ocean/hurricane/DEVR45to5rr1/init_subgrid/initial_state/' \
              'crm_vol1_2023_nc3.nc'
    geojson_file = 'map.geojson'

    variable_name = 'z'
    threshold = 0  # Adjust threshold based on your data
    xvar = 'lon'
    yvar = 'lat'

    centerlines, mask, X, Y = extract_longest_centerline_per_label(
        nc_file, geojson_file, topo_var=variable_name,
        threshold=threshold, x_var=xvar, y_var=yvar
    )

    spacing = .01
    offset_distance = 0.5 * spacing
    points = generate_offset_points_variable(centerlines,
                                             spacing, offset_distance)
    plot_linestrings(centerlines, mask, X, Y, points)


if __name__ == "__main__":
    main()
