import numpy as np
import netCDF4
import jigsawpy
import xarray
from matplotlib import pyplot as plt

from mpas_tools.mesh.creation import build_planar_mesh
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call

from compass.step import Step
from compass.model import make_graph_file


class Mesh(Step):
    """
    A step for creating a mesh and initial condition for thwaites test cases
    Attributes
    ----------
    mesh_type : str
        The resolution or mesh type of the test case
    """
    def __init__(self, test_case):
        """
        Create the step
        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        mesh_type : str
            The resolution or mesh type of the test case
        """
        super().__init__(test_case=test_case, name='mesh')

        self.add_output_file(filename='graph.info')
        self.add_output_file(filename='Thwaites_1to8km.nc')
        self.add_input_file(filename='antarctica_8km_2020_10_20.nc',
                            target='antarctica_8km_2020_10_20.nc',
                            database='')
        # Add geojson file to databse, download from github, or other?
        # Currently just using a local copy.
        self.add_input_file(filename='thwaites_minimal.geojson',
                            target='thwaites_minimal.geojson',
                            database=None, copy=True)
        self.add_input_file(filename='antarctica_1km_2020_10_20_ASE.nc',
                            target='antarctica_1km_2020_10_20_ASE.nc',
                            database='')

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
       """
        logger = self.logger
        try:
            config = self.config
            section = config['high_res_mesh']
        except KeyError:
            print('No config file; skipping')

        print('calling build_cell_width')
        cell_width, x1, y1, geom_points, geom_edges = self.build_cell_width()
        print('calling build_planar_mesh')
        build_planar_mesh(cell_width, x1, y1, geom_points,
                          geom_edges, logger=logger)
        dsMesh = xarray.open_dataset('base_mesh.nc')
        print('culling mesh')
        dsMesh = cull(dsMesh, logger=logger)
        print('converting to MPAS mesh')
        dsMesh = convert(dsMesh, logger=logger)
        print('writing grid_converted.nc')
        write_netcdf(dsMesh, 'grid_converted.nc')
        # If no number of levels specified in config file, use 10
        try:
            levels = section.get('levels')
        except NameError:
            levels = '10'
        print('calling create_landice_grid_from_generic_MPAS_grid.py')
        args = ['create_landice_grid_from_generic_MPAS_grid.py',
                '-i', 'grid_converted.nc',
                '-o', 'ase_1km_preCull.nc',
                '-l', levels, '-v', 'glimmer']
        check_call(args, logger=logger)

        print('calling interpolate_to_mpasli_grid.py')
        args = ['interpolate_to_mpasli_grid.py', '-s',
                'antarctica_1km_2020_10_20_ASE.nc', '-d',
                'ase_1km_preCull.nc', '-m', 'b', '-t']

        check_call(args, logger=logger)

        print('calling define_cullMask.py')
        args = ['define_cullMask.py', '-f',
                'ase_1km_preCull.nc', '-m'
                'distance', '-d', '50.0']

        check_call(args, logger=logger)

        print('calling set_lat_lon_fields_in_planar_grid.py')
        args = ['set_lat_lon_fields_in_planar_grid.py', '-f',
                'ase_1km_preCull.nc', '-p', 'ais-bedmap2']

        check_call(args, logger=logger)

        print('calling MpasMaskCreator.x')
        args = ['MpasMaskCreator.x', 'ase_1km_preCull.nc',
                'thwaites_mask.nc', '-f', 'thwaites_minimal.geojson']

        check_call(args, logger=logger)

        print('culling to geojson file')
        dsMesh = xarray.open_dataset('ase_1km_preCull.nc')
        thwaitesMask = xarray.open_dataset('thwaites_mask.nc')
        dsMesh = cull(dsMesh, dsInverse=thwaitesMask, logger=logger)
        write_netcdf(dsMesh, 'thwaites_culled.nc')

        print('Marking horns for culling')
        args = ['mark_horns_for_culling.py', '-f', 'thwaites_culled.nc']
        check_call(args, logger=logger)

        print('culling and converting')
        dsMesh = xarray.open_dataset('thwaites_culled.nc')
        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, logger=logger)
        write_netcdf(dsMesh, 'thwaites_dehorned.nc')

        print('calling create_landice_grid_from_generic_MPAS_grid.py')
        args = ['create_landice_grid_from_generic_MPAS_grid.py', '-i',
                'thwaites_dehorned.nc', '-o',
                'Thwaites_1to8km.nc', '-l', levels, '-v', 'glimmer',
                '--beta', '--thermal', '--obs', '--diri']

        check_call(args, logger=logger)

        print('calling interpolate_to_mpasli_grid.py')
        args = ['interpolate_to_mpasli_grid.py', '-s',
                'antarctica_1km_2020_10_20_ASE.nc',
                '-d', 'Thwaites_1to8km.nc', '-m', 'b', '-t']
        check_call(args, logger=logger)

        print('Marking domain boundaries dirichlet')
        args = ['mark_domain_boundaries_dirichlet.py',
                '-f', 'Thwaites_1to8km.nc']
        check_call(args, logger=logger)

        print('calling set_lat_lon_fields_in_planar_grid.py')
        args = ['set_lat_lon_fields_in_planar_grid.py', '-f',
                'Thwaites_1to8km.nc', '-p', 'ais-bedmap2']
        check_call(args, logger=logger)

        print('creating graph.info')
        make_graph_file(mesh_filename='Thwaites_1to8km.nc',
                        graph_filename='graph.info')

    def build_cell_width(self):

        # get needed fields from Antarctica dataset
        f = netCDF4.Dataset('antarctica_8km_2020_10_20.nc', 'r')
        f.set_auto_mask(False)  # disable masked arrays

        x1 = f.variables['x1'][:]
        y1 = f.variables['y1'][:]
        thk = f.variables['thk'][0, :, :]
        topg = f.variables['topg'][0, :, :]
        vx = f.variables['vx'][0, :, :]
        vy = f.variables['vy'][0, :, :]

        # subset data - optional
        step = 1
        x1 = x1[::step]
        y1 = y1[::step]
        thk = thk[::step, ::step]
        topg = topg[::step, ::step]
        vx = vx[::step, ::step]
        vy = vy[::step, ::step]

        dx = x1[1] - x1[0]  # assumed constant and equal in x and y
        nx = len(x1)
        ny = len(y1)

        sz = thk.shape

        # define extent of region to mesh
        xx0 = -1864434
        xx1 = -975432
        yy0 = -901349
        yy1 = 0
        geom_points = np.array([  # list of xy "node" coordinates
            ((xx0, yy0), 0),
            ((xx1, yy0), 0),
            ((xx1, yy1), 0),
            ((xx0, yy1), 0)],
            dtype=jigsawpy.jigsaw_msh_t.VERT2_t)

        geom_edges = np.array([    # list of "edges" between nodes
            ((0, 1), 0),
            ((1, 2), 0),
            ((2, 3), 0),
            ((3, 0), 0)],
            dtype=jigsawpy.jigsaw_msh_t.EDGE2_t)

        # flood fill to remove island, icecaps, etc.
        searchedMask = np.zeros(sz)
        floodMask = np.zeros(sz)
        iStart = sz[0] // 2
        jStart = sz[1] // 2
        floodMask[iStart, jStart] = 1

        neighbors = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

        lastSearchList = np.ravel_multi_index([[iStart], [jStart]],
                                              sz, order='F')

        # flood fill -------------------
        cnt = 0
        while len(lastSearchList) > 0:
            # print(cnt)
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

                        if thk[ii, jj] > 0.0:
                            floodMask[ii, jj] = 1  # mark as ice
                            # add to list of newly found  cells
                            newSearchList = np.append(newSearchList,
                                                      np.ravel_multi_index(
                                                          [[ii], [jj]], sz,
                                                          order='F')[0])
            lastSearchList = newSearchList
        # optional - plot flood mask
        # plt.pcolor(floodMask)
        # plt.show()

        # apply flood fill
        thk[floodMask == 0] = 0.0
        vx[floodMask == 0] = 0.0
        vy[floodMask == 0] = 0.0

        # make masks -------------------
        neighbors = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                              [1, 1], [-1, 1], [1, -1], [-1, -1]])

        iceMask = thk > 0.0
        # groundedMask = thk > (-1028.0 / 910.0 * topg)
        # floatingMask = np.logical_and(thk < (-1028.0 /
        #                               910.0 * topg), thk > 0.0)
        marginMask = np.zeros(sz, dtype='i')
        for n in neighbors:
            marginMask = np.logical_or(marginMask,
                                       np.logical_not(
                                           np.roll(iceMask, n, axis=[0, 1])))
        # where ice exists and neighbors non-ice locations
        marginMask = np.logical_and(marginMask, iceMask)
        # optional - plot mask
        # plt.pcolor(marginMask); plt.show()

        # calc dist to margin -------------------
        [XPOS, YPOS] = np.meshgrid(x1, y1)
        distToEdge = np.zeros(sz)

        # -- KEY PARAMETER: how big of a search 'box' (one-directional) to use.
        # Bigger number makes search slower, but if too small, the transition
        # zone could get truncated. Could automatically set this from maxDist
        # variables used in next section.)
        windowSize = 100.0e3
        # ---

        d = int(np.ceil(windowSize / dx))
        # print(windowSize, d)
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

            dist2Here = ((XPOS[np.ix_(irng, jrng)] - x1[j])**2 +
                         (YPOS[np.ix_(irng, jrng)] - y1[i])**2)**0.5
            dist2Here[marginMask[np.ix_(irng, jrng)] == 0] = maxdist
            distToEdge[i, j] = dist2Here.min()
        # optional - plot distance calculation
        # plt.pcolor(distToEdge/1000.0); plt.colorbar(); plt.show()

        # now create cell spacing function -------
        speed = (vx**2 + vy**2)**0.5
        lspd = np.log10(speed)
        # threshold
        # ls_min = 0
        # ls_max = 3
        # lspd(lspd<ls_min) = ls_min
        # lspd(lspd>ls_max) = ls_max

        # make dens fn mapping from log speed to cell spacing
        minSpac = 1.00
        maxSpac = 8.0
        highLogSpeed = 2.5
        lowLogSpeed = 0.75
        spacing = np.interp(lspd, [lowLogSpeed, highLogSpeed],
                            [maxSpac, minSpac], left=maxSpac, right=minSpac)
        spacing[thk == 0.0] = minSpac
        # plt.pcolor(spacing); plt.colorbar(); plt.show()

        # make dens fn mapping for dist to margin
        minSpac = 1.0
        maxSpac = 8.0
        highDist = 100.0 * 1000.0  # m
        lowDist = 10.0 * 1000.0
        spacing2 = np.interp(distToEdge, [lowDist, highDist],
                             [minSpac, maxSpac], left=minSpac, right=maxSpac)
        spacing2[thk == 0.0] = minSpac
        # plt.pcolor(spacing2); plt.colorbar(); plt.show()

        # merge two cell spacing methods
        cell_width = np.minimum(spacing, spacing2) * 1000.0
        # put coarse res far out in non-ice area to keep mesh smaller in the
        # part we are going to cull anyway (speeds up whole process)
        cell_width[np.logical_and(thk == 0.0,
                   distToEdge > 20.0e3)] = maxSpac * 1000.0
        # plt.pcolor(cell_width); plt.colorbar(); plt.show()

        # cell_width = 20000.0 * np.ones(thk.shape)

        return (cell_width.astype('float64'), x1.astype('float64'),
                y1.astype('float64'), geom_points, geom_edges)
