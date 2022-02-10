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
from compass.landice.mesh import gridded_flood_fill


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
        self.add_input_file(filename='thwaites_minimal.geojson',
                            package='compass.landice.tests.thwaites',
                            target='thwaites_minimal.geojson',
                            database=None)
        self.add_input_file(filename='antarctica_1km_2020_10_20_ASE.nc',
                            target='antarctica_1km_2020_10_20_ASE.nc',
                            database='')

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        section = config['high_res_mesh']

        logger.info('calling build_cell_width')
        cell_width, x1, y1, geom_points, geom_edges = self.build_cell_width()
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
                '-o', 'ase_1km_preCull.nc',
                '-l', levels, '-v', 'glimmer']
        check_call(args, logger=logger)

        # This step uses a subset of the whole Antarctica dataset trimmed to
        # the Amundsen Sean Embayment, to speed up interpolation.
        # This could also be replaced with the full Antarctic Ice Sheet
        # dataset.
        logger.info('calling interpolate_to_mpasli_grid.py')
        args = ['interpolate_to_mpasli_grid.py', '-s',
                'antarctica_1km_2020_10_20_ASE.nc', '-d',
                'ase_1km_preCull.nc', '-m', 'b', '-t']

        check_call(args, logger=logger)

        # This step is only necessary if you wish to cull a certain
        # distance from the ice margin, within the bounds defined by
        # the GeoJSON file.
        cullDistance = section.get('cullDistance')
        if float(cullDistance) > 0.:
            logger.info('calling define_cullMask.py')
            args = ['define_cullMask.py', '-f',
                    'ase_1km_preCull.nc', '-m'
                    'distance', '-d', cullDistance]

            check_call(args, logger=logger)
        else:
            logger.info('cullDistance <= 0 in config file. '
                        'Will not cull by distance to margin. \n')

        # This step is only necessary because the GeoJSON region
        # is defined by lat-lon.
        logger.info('calling set_lat_lon_fields_in_planar_grid.py')
        args = ['set_lat_lon_fields_in_planar_grid.py', '-f',
                'ase_1km_preCull.nc', '-p', 'ais-bedmap2']

        check_call(args, logger=logger)

        logger.info('calling MpasMaskCreator.x')
        args = ['MpasMaskCreator.x', 'ase_1km_preCull.nc',
                'thwaites_mask.nc', '-f', 'thwaites_minimal.geojson']

        check_call(args, logger=logger)

        logger.info('culling to geojson file')
        dsMesh = xarray.open_dataset('ase_1km_preCull.nc')
        thwaitesMask = xarray.open_dataset('thwaites_mask.nc')
        dsMesh = cull(dsMesh, dsInverse=thwaitesMask, logger=logger)
        write_netcdf(dsMesh, 'thwaites_culled.nc')

        logger.info('Marking horns for culling')
        args = ['mark_horns_for_culling.py', '-f', 'thwaites_culled.nc']
        check_call(args, logger=logger)

        logger.info('culling and converting')
        dsMesh = xarray.open_dataset('thwaites_culled.nc')
        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, logger=logger)
        write_netcdf(dsMesh, 'thwaites_dehorned.nc')

        logger.info('calling create_landice_grid_from_generic_MPAS_grid.py')
        args = ['create_landice_grid_from_generic_MPAS_grid.py', '-i',
                'thwaites_dehorned.nc', '-o',
                'Thwaites_1to8km.nc', '-l', levels, '-v', 'glimmer',
                '--beta', '--thermal', '--obs', '--diri']

        check_call(args, logger=logger)

        logger.info('calling interpolate_to_mpasli_grid.py')
        args = ['interpolate_to_mpasli_grid.py', '-s',
                'antarctica_1km_2020_10_20_ASE.nc',
                '-d', 'Thwaites_1to8km.nc', '-m', 'b', '-t']
        check_call(args, logger=logger)

        logger.info('Marking domain boundaries dirichlet')
        args = ['mark_domain_boundaries_dirichlet.py',
                '-f', 'Thwaites_1to8km.nc']
        check_call(args, logger=logger)

        logger.info('calling set_lat_lon_fields_in_planar_grid.py')
        args = ['set_lat_lon_fields_in_planar_grid.py', '-f',
                'Thwaites_1to8km.nc', '-p', 'ais-bedmap2']
        check_call(args, logger=logger)

        logger.info('creating graph.info')
        make_graph_file(mesh_filename='Thwaites_1to8km.nc',
                        graph_filename='graph.info')

    def build_cell_width(self):
        """
        Determine MPAS mesh cell size based on user-defined density function.

        This includes hard-coded definition of the extent of the regional
        mesh and user-defined mesh density functions based on observed flow
        speed and distance to the ice margin. In the future, this function
        and its components will likely be separated into separate generalized
        functions to be reusable by multiple test groups.
        """
        # get needed fields from Antarctica dataset
        f = netCDF4.Dataset('antarctica_8km_2020_10_20.nc', 'r')
        f.set_auto_mask(False)  # disable masked arrays
        logger = self.logger

        x1 = f.variables['x1'][:]
        y1 = f.variables['y1'][:]
        thk = f.variables['thk'][0, :, :]
        topg = f.variables['topg'][0, :, :]
        vx = f.variables['vx'][0, :, :]
        vy = f.variables['vy'][0, :, :]

        dx = x1[1] - x1[0]  # assumed constant and equal in x and y
        nx = len(x1)
        ny = len(y1)

        sz = thk.shape

        # Define extent of region to mesh.
        # These coords are specific to the Thwaites Glacier mesh.
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

        # Remove ice not connected to the ice sheet.
        floodMask = gridded_flood_fill(thk)
        thk[floodMask == 0] = 0.0
        vx[floodMask == 0] = 0.0
        vy[floodMask == 0] = 0.0

        # make masks -------------------
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

        # calc dist to margin -------------------
        [XPOS, YPOS] = np.meshgrid(x1, y1)
        distToEdge = np.zeros(sz)
        distToGroundingLine = np.zeros(sz)

        # -- KEY PARAMETER: how big of a search 'box' (one-directional) to use
        # to calculate the distance from each cell to the ice margin.
        # Bigger number makes search slower, but if too small, the transition
        # zone could get truncated. Could automatically set this from maxDist
        # variables used in next section. Currently, this is only used to
        # determine mesh spacing in ice-free areas in order to keep mesh
        # density low in areas that will be culled.
        windowSize = 100.0e3
        # ---

        d = int(np.ceil(windowSize / dx))
        # logger.info(windowSize, d)
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

            dist2HereEdge = dist2Here.copy()
            dist2HereGroundingLine = dist2Here.copy()

            dist2HereEdge[marginMask[np.ix_(irng, jrng)] == 0] = maxdist
            dist2HereGroundingLine[groundingLineMask
                                   [np.ix_(irng, jrng)] == 0] = maxdist

            distToEdge[i, j] = dist2HereEdge.min()
            distToGroundingLine[i, j] = dist2HereGroundingLine.min()
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
        minSpac = 1.0
        maxSpac = 8.0
        highLogSpeed = 2.5
        lowLogSpeed = 0.75
        spacing = np.interp(lspd, [lowLogSpeed, highLogSpeed],
                            [maxSpac, minSpac], left=maxSpac, right=minSpac)
        spacing[thk == 0.0] = minSpac
        # plt.pcolor(spacing); plt.colorbar(); plt.show()

        # make dens fn mapping for dist to ice margin
        minSpac = 1.0
        maxSpac = 8.0
        highDist = 100.0 * 1000.0  # m
        lowDist = 50.0 * 1000.0
        spacing2 = np.interp(distToEdge, [lowDist, highDist],
                             [minSpac, maxSpac], left=minSpac, right=maxSpac)
        spacing2[thk == 0.0] = minSpac
        # plt.pcolor(spacing2); plt.colorbar(); plt.show()

        # make dens fn mapping for dist to grounding line
        minSpac = 1.0
        maxSpac = 8.0
        highdist = 100.0 * 1000.0  # m
        lowDist = 50.0 * 1000.0
        spacing3 = np.interp(distToGroundingLine, [lowDist, highDist],
                             [minSpac, maxSpac], left=minSpac, right=maxSpac)
        spacing3[thk == 0.0] = minSpac
        # merge cell spacing methods
        cell_width = np.minimum(spacing, spacing2)
        cell_width = np.minimum(cell_width, spacing3) * 1000.0
        # put coarse res far out in non-ice area to keep mesh smaller in the
        # part we are going to cull anyway (speeds up whole process)
        cell_width[np.logical_and(thk == 0.0,
                   distToEdge > 25.0e3)] = maxSpac * 1000.0
        # plt.pcolor(cell_width); plt.colorbar(); plt.show()

        # cell_width = 20000.0 * np.ones(thk.shape)

        return (cell_width.astype('float64'), x1.astype('float64'),
                y1.astype('float64'), geom_points, geom_edges)
