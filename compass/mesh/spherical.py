import xarray
import xarray.plot
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import jigsawpy
from jigsawpy.savejig import savejig

from mpas_tools.cime.constants import constants
from mpas_tools.mesh.conversion import convert
from mpas_tools.mesh.creation.jigsaw_to_netcdf import jigsaw_to_netcdf
from mpas_tools.ocean.inject_meshDensity import inject_spherical_meshDensity
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from mpas_tools.viz.colormaps import register_sci_viz_colormaps
from mpas_tools.viz.paraview_extractor import extract_vtk

from compass.step import Step
from compass.model import make_graph_file


class SphericalBaseStep(Step):
    """
    A base class for steps that create a JIGSAW spherical mesh

    Attributes
    ----------
    opts : jigsawpy.jigsaw_jig_t
        JIGSAW options for creating the mesh
    """
    def __init__(self, test_case, name, subdir):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.testcase.TestCase
            The test case this step belongs to

        name : str
            the name of the step

        subdir : {str, None}
            the subdirectory for the step.  The default is ``name``
        """
        super().__init__(test_case, name=name, subdir=subdir)

        # setup files for JIGSAW
        self.opts = jigsawpy.jigsaw_jig_t()

    def save_and_plot_cell_width(self, lon, lat, cell_width):
        """
        Save the cell width field on a lon/lat grid to
        ``self.cell_width_filename`` and plot

        Parameters
        ----------
        lon : numpy.ndarray
            longitude in degrees (length n and between -180 and 180)

        lat : numpy.ndarray
            longitude in degrees (length m and between -90 and 90)

        cell_width : numpy.ndarray
            m x n array of cell width in km
        """
        section = self.config['spherical_mesh']
        da = xarray.DataArray(
            cell_width, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon},
            name='cellWidth')
        cell_width_filename = section.get('cell_width_filename')
        da.to_netcdf(cell_width_filename)

        if section.getboolean('plot_cell_width'):
            self._plot_cell_width(cell_width)

    def setup(self):
        """
        Add output files
        """
        config = self.config
        for option in ['jigsaw_mesh_filename', 'mpas_mesh_filename',
                       'cell_width_filename']:
            filename = config.get('spherical_mesh', option)
            self.add_output_file(filename=filename)
        self.add_output_file(filename='graph.info')

    def run(self):
        """
        Finish up the step. Run this after the subclass's run() method
        """
        logger = self.logger
        config = self.config
        section = config['spherical_mesh']
        earth_radius = constants['SHR_CONST_REARTH']
        jigsaw_mesh_filename = section.get('jigsaw_mesh_filename')
        mpas_mesh_filename = section.get('mpas_mesh_filename')

        logger.info('Convert triangles from jigsaw format to netcdf')
        jigsaw_to_netcdf(msh_filename=jigsaw_mesh_filename,
                         output_name='mesh_triangles.nc', on_sphere=True,
                         sphere_radius=earth_radius)

        logger.info('Convert from triangles to MPAS mesh')
        write_netcdf(convert(xarray.open_dataset('mesh_triangles.nc'),
                             dir='.', logger=logger),
                     mpas_mesh_filename)

        if section.getboolean('add_mesh_density'):
            logger.info(f'Add meshDensity into the mesh file')
            ds = xarray.open_dataset('cellWidthVsLatLon.nc')
            inject_spherical_meshDensity(
                ds.cellWidth.values, ds.lon.values, ds.lat.values,
                mesh_filename=mpas_mesh_filename)

        if section.getboolean('convert_to_vtk'):
            vtk_dir = section.get('vtk_dir')
            # only use progress bars if we're not writing to a log file
            use_progress_bar = self.log_filename is None
            lat_lon = section.getboolean('vtk_lat_lon')

            logger.info('Create vtk file for visualization')
            extract_vtk(ignore_time=True, lonlat=lat_lon,
                        dimension_list=['maxEdges='],
                        variable_list=['allOnCells'],
                        filename_pattern=mpas_mesh_filename,
                        out_dir=vtk_dir, use_progress_bar=use_progress_bar)

        make_graph_file(mesh_filename=mpas_mesh_filename,
                        graph_filename='graph.info')

    def _plot_cell_width(self, cell_width):
        """
        Plot a lat/lon map of cell widths (mesh resolution)

        Parameters
        ----------
        cell_width : numpy.ndarray
            m x n array of cell width in km
        """
        config = self.config
        cmap = config.get('spherical_mesh', 'cell_width_colormap')
        image_filename = config.get('spherical_mesh',
                                    'cell_width_image_filename')
        register_sci_viz_colormaps()
        fig = plt.figure(figsize=[16.0, 8.0])
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        im = ax.imshow(cell_width, origin='lower',
                       transform=ccrs.PlateCarree(),
                       extent=[-180, 180, -90, 90], cmap=cmap, zorder=0)
        ax.add_feature(cartopy.feature.LAND, edgecolor='black', zorder=1)
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=1,
            color='gray',
            alpha=0.5,
            linestyle='-', zorder=2)
        gl.top_labels = False
        gl.right_labels = False
        min_width = np.amin(cell_width)
        max_width = np.amax(cell_width)
        plt.title(
            f'Grid cell size, km, min: {min_width:.1f} max: {max_width:.1f}')
        plt.colorbar(im, shrink=.60)
        fig.canvas.draw()
        plt.tight_layout()
        plt.savefig(image_filename, bbox_inches='tight')
        plt.close()


class QuasiUniformSphericalMeshStep(SphericalBaseStep):
    """
    A step for creating a quasi-uniform JIGSAW mesh with a constant approximate
    cell width.  Subclasses can override the ``build_cell_width_lat_lon()``
    method to define a lon/lat map of cell width.  They can override the
    ``make_jigsaw_mesh()`` method to change how the jigsaw mesh is generated.

    Attributes
    ----------
    cell_width : float
        The approximate cell width in km of the mesh if constant resolution
    """

    def __init__(self, test_case, name='base_mesh', subdir=None,
                 cell_width=None):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.testcase.TestCase
            The test case this step belongs to

        name : str, optional
            the name of the step

        subdir : {str, None}, optional
            the subdirectory for the step

        cell_width : float, optional
            The approximate cell width in km of the mesh if constant resolution
        """
        super().__init__(test_case=test_case, name=name, subdir=subdir)
        self.cell_width = cell_width

        # build mesh via JIGSAW!
        self.opts.hfun_scal = 'absolute'
        self.opts.hfun_hmax = float('inf')
        self.opts.hfun_hmin = 0.0
        # 2-dim. simplexes
        self.opts.mesh_dims = 2
        self.opts.optm_qlim = 0.9375
        self.opts.verbosity = 1

    def setup(self):
        """
        Add JIGSAW options based on config options
        """
        section = self.config['spherical_mesh']
        self.opts.mesh_file = section.get('jigsaw_mesh_filename')
        self.opts.geom_file = section.get('jigsaw_geom_filename')
        self.opts.jcfg_file = section.get('jigsaw_jcfg_filename')
        self.opts.hfun_file = section.get('jigsaw_hfun_filename')
        super().setup()

    def run(self):
        """
        Run this step of the test case
        """
        cell_width, lon, lat = self.build_cell_width_lat_lon()
        self.save_and_plot_cell_width(lon, lat, cell_width)

        self.make_jigsaw_mesh(lon, lat, cell_width)

        # do the rest of the step (converting to the MPAS base mesh)
        super().run()

    def build_cell_width_lat_lon(self):
        """
        A function for creating cell width array for this mesh on a regular
        latitude-longitude grid.  The default method use constant resolution
        by setting the ``cell_width`` config option in the ``[spherical_mesh]``
        section. Subclasses should override this function if a more complex
        cell-width map is needed.

        Returns
        -------
        cell_width : numpy.ndarray
            m x n array of cell width in km

        lon : numpy.ndarray
            longitude in degrees (length n and between -180 and 180)

        lat : numpy.ndarray
            longitude in degrees (length m and between -90 and 90)
        """

        logger = self.logger
        cell_width = self.cell_width
        if cell_width is None:
            raise ValueError('The cell width was not set.')

        logger.info(f'  cell width: {cell_width} km')

        # save the constant approximate resolution on a 10 degree grid
        dlon = 10.
        dlat = dlon
        nlon = int(360./dlon) + 1
        nlat = int(180./dlat) + 1
        lon = np.linspace(-180., 180., nlon)
        lat = np.linspace(-90., 90., nlat)
        cell_width = cell_width * np.ones((nlat, nlon))
        return cell_width, lon, lat

    def make_jigsaw_mesh(self, lon, lat, cell_width):
        """
        Build the JIGSAW mesh.  A subclass can override this method to build
        the mesh in a different way.

        lon : numpy.ndarray
            longitude in degrees (length n and between -180 and 180)

        lat : numpy.ndarray
            longitude in degrees (length m and between -90 and 90)

        cell_width : numpy.ndarray
            m x n array of cell width in km
        """
        logger = self.logger
        earth_radius = constants['SHR_CONST_REARTH']
        opts = self.opts

        # save HFUN data to file
        hmat = jigsawpy.jigsaw_msh_t()
        hmat.mshID = 'ELLIPSOID-GRID'
        hmat.xgrid = np.radians(lon)
        hmat.ygrid = np.radians(lat)
        hmat.value = cell_width
        jigsawpy.savemsh(opts.hfun_file, hmat)

        # define JIGSAW geometry
        geom = jigsawpy.jigsaw_msh_t()
        geom.mshID = 'ELLIPSOID-MESH'
        geom.radii = earth_radius*1e-3*np.ones(3, float)
        jigsawpy.savemsh(opts.geom_file, geom)

        savejig(opts.jcfg_file, opts)
        check_call(['jigsaw', opts.jcfg_file], logger=logger)


class IcosahedralMeshStep(SphericalBaseStep):
    """
    A step for creating an icosahedral JIGSAW mesh

    Attributes
    ----------
    cell_width : float
        The approximate cell width in km of the mesh if constant resolution

    subdivisions : int
        The number of subdivisions of the icosahedral mesh to perform
    """

    def __init__(self, test_case, name='base_mesh', subdir=None,
                 cell_width=None, subdivisions=None):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.testcase.TestCase
            The test case this step belongs to

        name : str, optional
            the name of the step

        subdir : {str, None}, optional
            the subdirectory for the step

        cell_width : float, optional
            The approximate cell width in km of the mesh if constant resolution

        subdivisions : int, optional
            The number of subdivisions of the icosahedral mesh to perform
        """
        super().__init__(test_case=test_case, name=name, subdir=subdir)

        # run as a subprocess so output goes to a log file
        self.run_as_subprocess = True

        self.cell_width = cell_width
        self.subdivisions = subdivisions

        self.opts.hfun_hmax = 1.
        # 2-dim. simplexes
        self.opts.mesh_dims = 2

        self.opts.optm_iter = 512
        self.opts.optm_qtol = 1e-6

    def setup(self):
        """
        Add JIGSAW options based on config options
        """
        section = self.config['spherical_mesh']
        self.opts.mesh_file = section.get('jigsaw_mesh_filename')
        self.opts.geom_file = section.get('jigsaw_geom_filename')
        self.opts.jcfg_file = section.get('jigsaw_jcfg_filename')
        super().setup()

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger

        logger.info('Generate JIGSAW icosahedral mesh')

        subdivisions, cell_width, lon, lat = \
            self.build_subdivisions_cell_width_lat_lon()
        self.save_and_plot_cell_width(lon, lat, cell_width)

        self.make_jigsaw_mesh(subdivisions)
        # do the rest of the step (converting to the MPAS base mesh)
        super().run()

    def make_jigsaw_mesh(self, subdivisions):
        """
        Make the JIGSAW mesh.  A subclass can override this method to build
        the mesh in a different way.

        Parameters
        ----------
        subdivisions : int
            The number of subdivisions of the icosahedron
        """
        earth_radius = constants['SHR_CONST_REARTH']
        opts = self.opts

        geom = jigsawpy.jigsaw_msh_t()
        geom.mshID = 'ellipsoid-mesh'
        geom.radii = earth_radius*1e-3*np.ones(3, float)
        jigsawpy.savemsh(opts.geom_file, geom)

        icos = jigsawpy.jigsaw_msh_t()
        jigsawpy.cmd.icosahedron(opts, subdivisions, icos)

    def build_subdivisions_cell_width_lat_lon(self):
        """
        A function for creating cell width array for this mesh on a regular
        latitude-longitude grid.

        Returns
        -------
        subdivisions : int
            The number of subdivisions of the icosahedron to make

        cell_width : numpy.ndarray
            m x n array of cell width in km

        lon : numpy.ndarray
            longitude in degrees (length n and between -180 and 180)

        lat : numpy.ndarray
            longitude in degrees (length m and between -90 and 90)
        """
        logger = self.logger
        config = self.config
        icosahedral_method = config.get('spherical_mesh', 'icosahedral_method')
        if icosahedral_method == 'subdivisions':
            subdivisions = self.subdivisions
            if subdivisions is None:
                raise ValueError('The number of subdivisions was not set.')
        elif icosahedral_method == 'cell_width':
            cell_width = self.cell_width
            if cell_width is None:
                raise ValueError('The cell width was not set.')
            subdivisions = self.get_subdivisions(cell_width)
        else:
            raise ValueError(f'"icosahedral_method" must be either '
                             f'"cell_width" or the "subdivisions".  Got '
                             f'{icosahedral_method}.')

        cell_width = self.get_cell_width(subdivisions)

        logger.info(f'  cell width: {cell_width} km')
        logger.info(f'  subdivisions: {subdivisions}')

        # save the constant approximate resolution on a 1 degree grid
        nlon = 361
        nlat = 181
        lon = np.linspace(-180., 180., nlon)
        lat = np.linspace(-90., 90., nlat)
        cell_width = cell_width * np.ones((nlat, nlon))
        return subdivisions, cell_width, lon, lat

    @staticmethod
    def get_subdivisions(cell_width):
        """
        Find the number of subdivisions of an icosahedron to achieve a
        resolution as close as possible to ``cell_width``.

        Parameters
        ----------
        cell_width : float
            The approximate size in km of each cell in the resulting mesh.

        Returns
        -------
        subdivisions : int
            The number of subdivisions of the icosahedron
        """
        earth_radius = constants['SHR_CONST_REARTH']
        earth_area = 4 * np.pi * earth_radius ** 2

        # Using Euclidean, not spherical area, so not accurate for large cell
        # widths
        triangle_area = np.sqrt(3)/4.*(cell_width*1e3)**2
        triangle_count = earth_area/triangle_area
        subdivisions = 0.5*np.log2(triangle_count/20)
        subdivisions = max(0, int(np.round(subdivisions)))

        return subdivisions

    @staticmethod
    def get_cell_width(subdivisions):
        """
        Get the approximate cell width for an icosahedral mesh given either a
        number of subdivisions of the icosahedron.  As a rule of thumb:

        ==============  =================
         subdivisions    cell width (km)
        ==============  =================
        5               240
        6               120
        7               60
        8               30
        9               15
        10              7.5
        11              3.8
        12              1.9
        13              0.94
        ==============  =================

        Parameters
        ----------
        subdivisions : int
            The number of subdivisions of the icosahedron to make

        Returns
        -------
        cell_width : float
            The approximate size in km of each cell
        """
        earth_radius = constants['SHR_CONST_REARTH']
        earth_area = 4 * np.pi * earth_radius ** 2

        # compute and save the cell widths for later use in computing mesh
        # density
        triangle_count = 20*2**(2*subdivisions)
        triangle_area = earth_area/triangle_count
        cell_width = 1e-3*np.sqrt(triangle_area*4./np.sqrt(3))
        return cell_width
