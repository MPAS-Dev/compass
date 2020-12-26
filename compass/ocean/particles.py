import netCDF4
import numpy as np
from pyamg.classical import interpolate as amginterp
from pyamg.classical import split
from scipy import sparse, spatial


VERTICAL_TREATMENTS = {"indexLevel": 1,
                       "fixedZLevel": 2,
                       "passiveFloat": 3,
                       "buoyancySurface": 4,
                       "argoFloat": 5}
DEFAULTS = {"dt": 300, "resettime": 1.0 * 24.0 * 60.0 * 60.0}
TYPELIST = ["buoyancy", "passive", "surface", "all"]
VERTSEEDTYPE = ["linear", "denseCenter", "log"]
SPATIAL_FILTER = ["SouthernOceanPlanar", "SouthernOceanXYZ"]


def write(init_filename, graph_filename, particle_filename, types='all',
          n_vert_levels=10, vert_seed_type='linear', n_buoy_surf=11,
          pot_dens_min=1028.5, pot_dens_max=1030.0, spatial_filter=None,
          downsample=0, seed_center=True, seed_vertex=False,
          add_noise=False, cfl_min=0.005):
    """
    Write an initial condition for particles partitioned across cores

    Parameters
    ----------
    init_filename : str
        path of netCDF init/mesh file

    graph_filename : str
        path of graph partition file of form */*.info.part

    particle_filename : str
        path of output netCDF particle file

    types : {"buoyancy", "passive", "surface", "all"}, optional
        types of particles",

    n_vert_levels : int, optional
        number of vertical levels for passive, 3D floats

    vert_seed_type : {"linear", "denseCenter", "log"}, optional
        method for seeding in the vertical

    n_buoy_surf : int, optional
        number of buoyancy surfaces for isopycnally-constrained particles

    pot_dens_min : float, optional
        minimum value of potential density surface for isopycnally-constrained
        particles

    pot_dens_max : float, optional
        maximum value of potential density surface for isopycnally-constrained
        particles

    spatial_filter : {"SouthernOceanPlanar", "SouthernOceanXYZ"}, optional
        apply a certain type of spatial filter

    downsample : int, optional
        downsample particle positions using AMG a number of times

    seed_center : bool, optional
        seed particles on cell centers

    seed_vertex : bool, optional
        seed three particles by a fixed epsilon off each cell vertex

    add_noise : bool, optional
        add gaussian noise to generate three particles around the cell center

    cfl_min : float, optional
        minimum assumed CFL, which is used in perturbing particles if
        ``seed_vertex=True`` or ``add_noise=True``
    """

    buoy_surf = np.linspace(pot_dens_min, pot_dens_max, n_buoy_surf)
    cpts, xCell, yCell, zCell = _particle_coords(
        init_filename, downsample, seed_center, seed_vertex, add_noise,
        cfl_min)

    # build particles
    particlelist = []
    if "buoyancy" in types or "all" in types:
        particlelist.append(_build_isopycnal_particles(
            cpts, xCell, yCell, zCell, buoy_surf, spatial_filter))
    if "passive" in types or "all" in types:
        particlelist.append(_build_passive_floats(
            cpts, xCell, yCell, zCell, init_filename, n_vert_levels,
            spatial_filter, vert_seed_type))
    # apply surface particles everywhere to ensure that LIGHT works
    # (allow for some load-imbalance for filters)
    if "surface" in types or "all" in types:
        particlelist.append(_build_surface_floats(
            cpts, xCell, yCell, zCell, spatial_filter))

    # write particles to disk
    ParticleList(particlelist).write(particle_filename, graph_filename)


def remap_particles(init_filename, particle_filename, graph_filename):
    """
    Remap particles onto a new grid decomposition.

    Load in particle positions, locations of grid cell centers, and
    decomposition corresponding to ``init_filename``.

    The goal is to update particle field ``currentBlock`` to comply with the
    new grid as defined by ``init_filename`` and ``particle_filename``.
    NOTE: ``init_filename`` and ``graph_filename`` must be compatible!

    We assume that all particles will be within the domain such that a nearest
    neighbor search is sufficient to make the remap.

    Parameters
    ----------
    init_filename : str
        path of netCDF init/mesh file

    graph_filename : str
        path of graph partition file of form */*.info.part

    particle_filename : str
        path of output netCDF particle file
    """
    # load the files
    with netCDF4.Dataset(init_filename, "r") as f_in, \
            netCDF4.Dataset(graph_filename, "r+") as f_part:

        # get the particle data
        xpart = f_part.variables["xParticle"]
        ypart = f_part.variables["yParticle"]
        zpart = f_part.variables["zParticle"]
        currentBlock = f_part.variables["currentBlock"]
        try:
            currentCell = f_part.variables["currentCell"]
            currentCellGlobalID = f_part.variables["currentCellGlobalID"]
        except KeyError:
            currentCell = f_part.createVariable("currentCell", "i",
                                                ("nParticles",))
            currentCellGlobalID = f_part.createVariable(
                "currentCellGlobalID", "i", ("nParticles",))

        # get the cell positions
        xcell = f_in.variables["xCell"]
        ycell = f_in.variables["yCell"]
        zcell = f_in.variables["zCell"]

        # build the spatial tree
        tree = spatial.cKDTree(np.vstack((xcell, ycell, zcell)).T)

        # get nearest cell for each particle
        dvEdge = f_in.variables["dvEdge"]
        maxdist = 2.0 * max(dvEdge[:])
        _, cellIndices = tree.query(
            np.vstack((xpart, ypart, zpart)).T, distance_upper_bound=maxdist,
            k=1)

        # load the decomposition (apply to latest time step)
        decomp = np.genfromtxt(particle_filename)
        currentBlock[-1, :] = decomp[cellIndices]
        currentCell[-1, :] = -1
        currentCellGlobalID[-1, :] = cellIndices + 1


def _use_defaults(name, val):
    if (val is not None) or (val is not np.nan):
        return val
    else:
        return DEFAULTS[name]


def _ensure_shape(start, new):
    if isinstance(new, (int, float)):
        new *= np.ones_like(start)
    return new


def _southern_ocean_only_xyz(x, y, z, maxNorth=-45.0):
    sq = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lat = np.arcsin(z / sq)
    ok = np.pi / 180.0 * maxNorth
    ids = lat < ok
    return ids


def _southern_ocean_only_planar(x, y, z, maxy=1000.0 * 1e3):
    ids = y < maxy
    return ids


def _downsample_points(x, y, z, tri, nsplit):
    """
    Downsample points using algebraic multigrid splitting.

    Note, currently assumes that all points on grid are equidistant, which does
    a numeric (not area-weighted) downsampling.

    Phillip Wolfram
    LANL
    Origin: 03/09/2015, Updated: 01/14/2019
    """
    # reference on cleanest way to do this calculation:
    # https://www.mathworks.com/matlabcentral/answers/
    # 369143-how-to-do-delaunay-triangulation-and-return-an-adjacency-matrix

    # allocate the memory
    Np = x.shape[0]
    A = sparse.lil_matrix((Np, Np))

    # cleanup impartial cells (don't include the triangles on boundary)
    tri = tri[np.logical_not(np.any(tri == -1, axis=1)), :]

    # handle one direction for triangles
    A[tri[:, 0], tri[:, 1]] = 1
    A[tri[:, 1], tri[:, 2]] = 1
    A[tri[:, 2], tri[:, 0]] = 1

    # handle other direction (bi-directional graph)
    A[tri[:, 1], tri[:, 0]] = 1
    A[tri[:, 2], tri[:, 1]] = 1
    A[tri[:, 0], tri[:, 2]] = 1

    A = A.tocsr()

    Cpts = np.arange(Np)
    # Grab root-nodes (i.e., Coarse / Fine splitting)
    for ii in np.arange(nsplit):
        splitting = split.PMIS(A)
        # convert to index for subsetting particles
        Cpts = Cpts[np.asarray(splitting, dtype=bool)]

        if ii < nsplit - 1:
            P = amginterp.direct_interpolation(A, A, splitting)
            R = P.T.tocsr()
            A = R * A * P

    return Cpts, x[Cpts], y[Cpts], z[Cpts]


class Particles:
    def __init__(
        self,
        x,
        y,
        z,
        cellindices,
        verticaltreatment,
        dt=np.nan,
        zlevel=np.nan,
        indexlevel=np.nan,
        buoypart=np.nan,
        buoysurf=None,
        spatialfilter=None,
        resettime=np.nan,
        xreset=np.nan,
        yreset=np.nan,
        zreset=np.nan,
        zlevelreset=np.nan,
    ):

        # start with all the indices and restrict
        ids = np.ones((len(x)), dtype=bool)
        if type(spatialfilter) is str:
            spatialfilter = [spatialfilter]
        if spatialfilter:
            if np.max(["SouthernOceanXYZ" == afilter for afilter in
                       spatialfilter]):
                ids = np.logical_and(ids, _southern_ocean_only_xyz(x, y, z))
            if np.max(["SouthernOceanPlanar" == afilter for afilter in
                       spatialfilter]):
                ids = np.logical_and(ids, _southern_ocean_only_planar(x, y, z))

        self.x = x[ids]
        self.y = y[ids]
        self.z = z[ids]
        self.verticaltreatment = _ensure_shape(
            self.x, VERTICAL_TREATMENTS[verticaltreatment])
        self.nparticles = len(self.x)

        self.dt = dt

        # 3D passive floats
        self.zlevel = _ensure_shape(x, zlevel)[ids]

        # isopycnal floats
        if buoysurf is not None:
            self.buoysurf = buoysurf
        self.buoypart = _ensure_shape(x, buoypart)[ids]
        self.cellindices = cellindices[ids]
        self.cellGlobalID = cellindices[ids]

        # index level following floats
        self.indexlevel = _ensure_shape(x, indexlevel)[ids]

        # reset features
        self.resettime = _ensure_shape(x, resettime)[ids]
        self.xreset = _ensure_shape(x, xreset)[ids]
        self.yreset = _ensure_shape(x, yreset)[ids]
        self.zreset = _ensure_shape(x, zreset)[ids]
        self.zlevelreset = _ensure_shape(x, zlevelreset)[ids]

    def compute_lat_lon(self):
        """
        Ripped out whole-sale from latlon_coordinate_transforms.py
        PJW 01/15/2019
        """

        x = self.x
        y = self.y
        z = self.z

        self.latParticle = np.arcsin(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
        self.lonParticle = np.arctan2(y, x)


class ParticleList:
    def __init__(self, particlelist):
        self.particlelist = particlelist

    def aggregate(self):
        self.len()

        # buoyancysurf
        buoysurf = np.array([])
        for alist in self.particlelist:
            if "buoysurf" in dir(alist):
                buoysurf = np.unique(
                    np.setdiff1d(np.append(buoysurf, alist.buoysurf), None)
                )
        if len(buoysurf) > 0:
            self.buoysurf = np.asarray(buoysurf, dtype="f8")
        else:
            self.buoysurf = None

    def __getattr__(self, name):
        # __getattr__ ensures self.x is concatenated properly
        return self.concatenate(name)

    def concatenate(self, varname):
        var = getattr(self.particlelist[0], varname)
        for alist in self.particlelist[1:]:
            var = np.append(var, getattr(alist, varname))
        return var

    def append(self, particlelist):
        self.particlelist.append(particlelist[:])

    def len(self):
        self.nparticles = 0
        for alist in self.particlelist:
            self.nparticles += alist.nparticles

        return self.nparticles

    # probably a cleaner way to have this "fall through" to the particle
    # instances themselves, but didn't have time to sort this all out so this
    # isn't general for now
    def compute_lat_lon(self):
        for alist in self.particlelist:
            alist.compute_lat_lon()

    def write(self, f_name, f_decomp):

        decomp = np.genfromtxt(f_decomp)

        self.aggregate()
        assert (
            max(decomp) < self.nparticles
        ), "Number of particles must be larger than decomposition!"

        f_out = netCDF4.Dataset(f_name, "w", format="NETCDF3_64BIT_OFFSET")

        f_out.createDimension("Time")
        f_out.createDimension("nParticles", self.nparticles)

        f_out.createVariable("xParticle", "f8", ("Time", "nParticles"))
        f_out.createVariable("yParticle", "f8", ("Time", "nParticles"))
        f_out.createVariable("zParticle", "f8", ("Time", "nParticles"))
        f_out.createVariable("lonParticle", "f8", ("Time", "nParticles"))
        f_out.createVariable("latParticle", "f8", ("Time", "nParticles"))
        f_out.createVariable("zLevelParticle", "f8", ("Time", "nParticles"))
        f_out.createVariable("dtParticle", "f8", ("Time", "nParticles"))
        f_out.createVariable("buoyancyParticle", "f8", ("Time", "nParticles"))
        f_out.createVariable("currentBlock", "i", ("Time", "nParticles"))
        f_out.createVariable("currentCell", "i", ("Time", "nParticles"))
        f_out.createVariable("currentCellGlobalID", "i", ("Time",
                                                          "nParticles"))
        f_out.createVariable("indexToParticleID", "i", ("nParticles",))
        f_out.createVariable("verticalTreatment", "i", ("Time", "nParticles"))
        f_out.createVariable("indexLevel", "i", ("Time", "nParticles"))
        f_out.createVariable("resetTime", "i", ("nParticles",))
        f_out.createVariable("currentBlockReset", "i", ("nParticles",))
        f_out.createVariable("currentCellReset", "i", ("nParticles",))
        f_out.createVariable("xParticleReset", "f8", ("nParticles",))
        f_out.createVariable("yParticleReset", "f8", ("nParticles",))
        f_out.createVariable("zParticleReset", "f8", ("nParticles",))
        f_out.createVariable("zLevelParticleReset", "f8", ("nParticles",))

        f_out.variables["xParticle"][0, :] = self.x
        f_out.variables["yParticle"][0, :] = self.y
        f_out.variables["zParticle"][0, :] = self.z

        self.compute_lat_lon()
        f_out.variables["lonParticle"][0, :] = self.lonParticle
        f_out.variables["latParticle"][0, :] = self.latParticle

        f_out.variables["verticalTreatment"][0, :] = self.verticaltreatment

        f_out.variables["zLevelParticle"][0, :] = self.zlevel

        if self.buoysurf is not None and len(self.buoysurf) > 0:
            f_out.createDimension("nBuoyancySurfaces", len(self.buoysurf))
            f_out.createVariable("buoyancySurfaceValues", "f8",
                                 ("nBuoyancySurfaces"))
            f_out.variables["buoyancyParticle"][0, :] = self.buoypart
            f_out.variables["buoyancySurfaceValues"][:] = self.buoysurf

        f_out.variables["dtParticle"][0, :] = DEFAULTS["dt"]
        # assume single-processor mode for now
        f_out.variables["currentBlock"][:] = 0
        # reset each day
        f_out.variables["resetTime"][:] = DEFAULTS["resettime"]
        f_out.variables["indexLevel"][:] = 1
        f_out.variables["indexToParticleID"][:] = np.arange(self.nparticles)

        # resets
        f_out.variables["currentBlock"][0, :] = decomp[self.cellindices]
        f_out.variables["currentBlockReset"][:] = decomp[self.cellindices]
        f_out.variables["currentCell"][0, :] = -1
        f_out.variables["currentCellGlobalID"][0, :] = self.cellGlobalID + 1
        f_out.variables["currentCellReset"][:] = -1
        f_out.variables["xParticleReset"][:] = \
            f_out.variables["xParticle"][0, :]
        f_out.variables["yParticleReset"][:] = \
            f_out.variables["yParticle"][0, :]
        f_out.variables["zParticleReset"][:] = \
            f_out.variables["zParticle"][0, :]
        f_out.variables["zLevelParticleReset"][:] = \
            f_out.variables["zLevelParticle"][0, :]

        f_out.close()


def _rescale_for_shell(f_init, x, y, z):
    rearth = f_init.sphere_radius
    r = np.sqrt(x * x + y * y + z * z)
    x *= rearth / r
    y *= rearth / r
    z *= rearth / r
    return x, y, z


def _get_particle_coords(f_init, seed_center=True, seed_vertex=False,
                         add_noise=False, CFLmin=None):
    xCell = f_init.variables["xCell"][:]
    yCell = f_init.variables["yCell"][:]
    zCell = f_init.variables["zCell"][:]

    # Case of only cell-center seeding a single particle.
    if seed_center and not add_noise:
        cells_center = (xCell, yCell, zCell)
        cpts_center = np.arange(len(xCell))

    # Case of cell-center seeding with 3 particles distributed around the
    # center by noise.
    elif seed_center and add_noise:
        cellsOnCell = f_init.variables["cellsOnCell"][:, :]

        nCells = len(f_init.dimensions["nCells"])
        perturbation = CFLmin * np.ones((nCells,))

        allx = []
        ally = []
        allz = []
        allcpts = []
        # There are six potential cell neighbors to perturb the particles for.
        # This selects three random directions (without replacement) at every
        # cell.
        cellDirs = np.stack(
            [
                np.random.choice(np.arange(6), size=3, replace=False)
                for _ in range(nCells)
            ]
        )
        for ci in np.arange(3):
            epsilon = np.abs(np.random.normal(size=nCells))
            epsilon /= epsilon.max()
            # Adds gaussian noise at each cell, creating range of
            # [CFLMin, 2*CFLMin]
            theta = perturbation * epsilon + perturbation

            x = (1.0 - theta) * xCell + theta * xCell[
                cellsOnCell[range(nCells), cellDirs[:, ci]] - 1
            ]
            y = (1.0 - theta) * yCell + theta * yCell[
                cellsOnCell[range(nCells), cellDirs[:, ci]] - 1
            ]
            z = (1.0 - theta) * zCell + theta * zCell[
                cellsOnCell[range(nCells), cellDirs[:, ci]] - 1
            ]

            x, y, z = _rescale_for_shell(f_init, x, y, z)

            allx.append(x)
            ally.append(y)
            allz.append(z)
            allcpts.append(cellsOnCell[:, ci] - 1)
        cells_center = (
            np.concatenate(allx),
            np.concatenate(ally),
            np.concatenate(allz),
        )
        cpts_center = np.concatenate(allcpts)

    # Case of seeding 3 particles by a small epsilon around the vertices.
    if seed_vertex:
        cellsOnVertex = f_init.variables["cellsOnVertex"][:, :]
        xVertex = f_init.variables["xVertex"][:]
        yVertex = f_init.variables["yVertex"][:]
        zVertex = f_init.variables["zVertex"][:]

        nVertices = len(f_init.dimensions["nVertices"])
        perturbation = CFLmin * np.ones((nVertices,))

        allx = []
        ally = []
        allz = []
        allcpts = []
        for vi in np.arange(3):
            ids = np.where(cellsOnVertex[:, vi] != 0)[0]
            theta = perturbation[ids]

            x = (1.0 - theta) * xVertex[ids] + \
                theta * xCell[cellsOnVertex[ids, vi] - 1]
            y = (1.0 - theta) * yVertex[ids] + \
                theta * yCell[cellsOnVertex[ids, vi] - 1]
            z = (1.0 - theta) * zVertex[ids] + \
                theta * zCell[cellsOnVertex[ids, vi] - 1]

            x, y, z = _rescale_for_shell(f_init, x, y, z)

            allx.append(x)
            ally.append(y)
            allz.append(z)
            allcpts.append(cellsOnVertex[ids, vi] - 1)
        cells_vertex = (
            np.concatenate(allx),
            np.concatenate(ally),
            np.concatenate(allz),
        )
        cpts_vertex = np.concatenate(allcpts)

    # Allows for both cell-center and cell-vertex seeding.
    if seed_center and not seed_vertex:
        cells = cells_center
        cpts = cpts_center
    elif not seed_center and seed_vertex:
        cells = cells_vertex
        cpts = cpts_vertex
    else:
        cpts = np.concatenate((cpts_vertex, cpts_center))
        cells = (
            np.concatenate((cells_vertex[0], cells_center[0])),
            np.concatenate((cells_vertex[1], cells_center[1])),
            np.concatenate((cells_vertex[2], cells_center[2])),
        )
    return cells, cpts


def _expand_nlevels(x, n):
    return np.tile(x, (n))


def _particle_coords(
    f_init, downsample, seed_center, seed_vertex, add_noise, CFLmin
):

    f_init = netCDF4.Dataset(f_init, "r")
    cells, cpts = _get_particle_coords(
        f_init, seed_center, seed_vertex, add_noise, CFLmin
    )
    xCell, yCell, zCell = cells
    if downsample:
        tri = f_init.variables["cellsOnVertex"][:, :] - 1
        cpts, xCell, yCell, zCell = _downsample_points(
            xCell, yCell, zCell, tri, downsample
        )
    f_init.close()

    return cpts, xCell, yCell, zCell


def _build_isopycnal_particles(cpts, xCell, yCell, zCell, buoysurf, afilter):

    nparticles = len(xCell)
    nbuoysurf = buoysurf.shape[0]

    x = _expand_nlevels(xCell, nbuoysurf)
    y = _expand_nlevels(yCell, nbuoysurf)
    z = _expand_nlevels(zCell, nbuoysurf)

    buoypart = (
        (np.tile(buoysurf, (nparticles, 1)))
        .reshape(nparticles * nbuoysurf, order="F")
        .copy())
    cellindices = np.tile(cpts, (nbuoysurf))

    return Particles(x, y, z, cellindices, "buoyancySurface",
                     buoypart=buoypart, buoysurf=buoysurf,
                     spatialfilter=afilter)


def _build_passive_floats(cpts, xCell, yCell, zCell, f_init, nvertlevels,
                          afilter, vertseedtype):

    x = _expand_nlevels(xCell, nvertlevels)
    y = _expand_nlevels(yCell, nvertlevels)
    z = _expand_nlevels(zCell, nvertlevels)
    f_init = netCDF4.Dataset(f_init, "r")
    if vertseedtype == "linear":
        wgts = np.linspace(0, 1, nvertlevels + 2)[1:-1]
    elif vertseedtype == "log":
        wgts = np.geomspace(1.0 / (nvertlevels - 1), 1, nvertlevels + 1)[0:-1]
    elif vertseedtype == "denseCenter":
        wgts = _dense_center_seeding(nvertlevels)
    else:
        raise ValueError(
            "Must designate `vertseedtype` as one of the following: "
            + f"{VERTSEEDTYPE}"
        )
    zlevel = -np.kron(wgts, f_init.variables["bottomDepth"][cpts])
    cellindices = np.tile(cpts, (nvertlevels))
    f_init.close()

    return Particles(
        x, y, z, cellindices, "passiveFloat", zlevel=zlevel,
        spatialfilter=afilter)


def _dense_center_seeding(nVert):
    """
    Distributes passive floats with 50% of them occurring between 40% and 60%
    of the bottom depth.
    """
    nMid = np.ceil((1 / 2) * nVert)
    nRem = nVert - nMid
    if nRem % 2 != 0:
        nMid += 1
        nRem -= 1
    upper = np.linspace(0, 0.4, (int(nRem) // 2) + 1)
    center = np.linspace(0.4, 0.6, int(nMid) + 2)
    lower = np.linspace(0.6, 1, (int(nRem) // 2) + 1)
    c_wgts = np.concatenate([upper[1:], center[1:-1], lower[0:-1]])
    return c_wgts


def _build_surface_floats(cpts, xCell, yCell, zCell, afilter):

    x = _expand_nlevels(xCell, 1)
    y = _expand_nlevels(yCell, 1)
    z = _expand_nlevels(zCell, 1)
    cellindices = cpts

    return Particles(x, y, z, cellindices, "indexLevel", indexlevel=1,
                     zlevel=0, spatialfilter=afilter)


def _build_particle_file(f_init, f_name, f_decomp, types, spatialfilter,
                         buoySurf, nVertLevels, downsample, vertseedtype,
                         seed_center, seed_vertex, add_noise, CFLmin):

    cpts, xCell, yCell, zCell = _particle_coords(
        f_init, downsample, seed_center, seed_vertex, add_noise, CFLmin)

    # build particles
    particlelist = []
    if "buoyancy" in types or "all" in types:
        particlelist.append(
            _build_isopycnal_particles(
                cpts, xCell, yCell, zCell, buoySurf, spatialfilter))
    if "passive" in types or "all" in types:
        particlelist.append(
            _build_passive_floats(
                cpts, xCell, yCell, zCell, f_init, nVertLevels, spatialfilter,
                vertseedtype))
    # apply surface particles everywhere to ensure that LIGHT works
    # (allow for some load-imbalance for filters)
    if "surface" in types or "all" in types:
        particlelist.append(
            _build_surface_floats(cpts, xCell, yCell, zCell, spatialfilter))

    # write particles to disk
    ParticleList(particlelist).write(f_name, f_decomp)
