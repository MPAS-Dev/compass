import timeit

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import jigsawpy
import matplotlib.pyplot as plt
import numpy as np
from mpas_tools.cime.constants import constants
from netCDF4 import Dataset
from scipy import interpolate, spatial

from compass.mesh import QuasiUniformSphericalMeshStep


class WavesBaseMesh(QuasiUniformSphericalMeshStep):
    """
    A step for creating wave mesh based on an ocean mesh
    """
    def __init__(self, test_case, ocean_mesh, name='base_mesh', subdir=None):

        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         cell_width=None)

        mesh_path = ocean_mesh.steps['initial_state'].path
        self.add_input_file(
            filename='ocean_mesh.nc',
            work_dir_target=f'{mesh_path}/initial_state.nc')

    def setup(self):

        super().setup()
        self.opts.init_file = 'init.msh'

    def build_cell_width_lat_lon(self):
        """
        Create cell width array for this mesh on a regular latitude-longitude
        grid

        Returns
        -------
        cellWidth : numpy.array
            m x n array of cell width in km

        lon : numpy.array
            longitude in degrees (length n and between -180 and 180)

        lat : numpy.array
            longitude in degrees (length m and between -90 and 90)
        """
        km = 1000.0

        lon_min = -180.0
        lon_max = 180.0
        dlon = self.config.getfloat('wave_mesh', 'hfun_grid_spacing')
        nlon = int((lon_max - lon_min) / dlon) + 1
        lat_min = -90.0
        lat_max = 90.0
        nlat = int((lat_max - lat_min) / dlon) + 1

        xlon = np.linspace(lon_min, lon_max, nlon)
        ylat = np.linspace(lat_min, lat_max, nlat)

        earth_radius = constants['SHR_CONST_REARTH'] / km
        cell_width = self.cell_widthVsLatLon(xlon, ylat,
                                             earth_radius, 'ocean_mesh.nc')
        cell_width = cell_width / km

        self.create_initial_points('ocean_mesh.nc', xlon, ylat, cell_width,
                                   earth_radius, self.opts.init_file)

        hfun_slope_lim = self.config.getfloat('wave_mesh', 'hfun_slope_lim')
        cell_width = self.limit_spacing_gradient(xlon, ylat, cell_width,
                                                 earth_radius, hfun_slope_lim)

        return cell_width, xlon, ylat

    def cell_widthVsLatLon(self, lon, lat, sphere_radius, ocean_mesh):

        config = self.config

        depth_threshold_refined = config.getfloat('wave_mesh',
                                                  'depth_threshold_refined')
        dist_threshold_refined = config.getfloat('wave_mesh',
                                                 'distance_threshold_refined')
        depth_threshold_global = config.getfloat('wave_mesh',
                                                 'depth_threshold_global')
        dist_threshold_global = config.getfloat('wave_mesh',
                                                'distance_threshold_global')
        refined_res = config.getfloat('wave_mesh', 'refined_res')
        maxres = config.getfloat('wave_mesh', 'maxres')

        # Create structrued grid points
        nlon = lon.size
        nlat = lat.size
        Lon_grd, Lat_grd = np.meshgrid(np.radians(lon), np.radians(lat))
        xy_pts = np.vstack((Lon_grd.ravel(), Lat_grd.ravel())).T

        # Create structured grid points and background cell_with array
        cell_width = np.zeros(Lon_grd.shape) + maxres

        # Get ocean mesh variables
        nc_file = Dataset(ocean_mesh, 'r')
        areaCell = nc_file.variables['areaCell'][:]
        lonCell = nc_file.variables['lonCell'][:]
        latCell = nc_file.variables['latCell'][:]
        bottomDepth = nc_file.variables['bottomDepth'][:]

        # Transform 0,360 range to -180,180
        idx, = np.where(lonCell > np.pi)
        lonCell[idx] = lonCell[idx] - 2.0 * np.pi
        idx, = np.where(lonCell < -np.pi)
        lonCell[idx] = lonCell[idx] + 2.0 * np.pi

        # Interpolate cellWidth onto background grid
        cellWidth = 2.0 * np.sqrt(areaCell / np.pi)
        hfun = interpolate.NearestNDInterpolator((lonCell, latCell), cellWidth)
        hfun_interp = hfun(xy_pts)
        hfun_grd = np.reshape(hfun_interp, (nlat, nlon))

        # Interpolate bathymetry onto background grid
        bathy = interpolate.NearestNDInterpolator((lonCell, latCell),
                                                  bottomDepth)
        bathy_interp = bathy(xy_pts)
        bathy_grd = np.reshape(bathy_interp, (nlat, nlon))

        # Get distance to coasts
        D = self.distance_to_shapefile_points(lon, lat,
                                              sphere_radius, reggrid=True)

        # Apply refined region criteria
        idxx, idxy = np.where((bathy_grd < depth_threshold_refined) &
                              (bathy_grd > 0.0) &
                              (D < dist_threshold_refined) &
                              (hfun_grd < refined_res))
        cell_width[idxx, idxy] = hfun_grd[idxx, idxy]

        # Apply global region criteria
        idxx, idxy = np.where((bathy_grd < depth_threshold_global) &
                              (bathy_grd > 0.0) &
                              (D < dist_threshold_global))
        cell_width[idxx, idxy] = hfun_grd[idxx, idxy]

        # Plot
        fig = plt.figure(figsize=(16, 8))

        ax1 = fig.add_subplot(4, 1, 1, projection=ccrs.PlateCarree())
        plt1 = ax1.contourf(lon, lat, cell_width, transform=ccrs.PlateCarree())
        ax1.set_title('waves mesh cell width')
        fig.colorbar(plt1, ax=ax1)

        ax2 = fig.add_subplot(4, 1, 2, projection=ccrs.PlateCarree())
        plt2 = ax2.contourf(lon, lat, hfun_grd, transform=ccrs.PlateCarree())
        ax2.set_title('ocean cell width')
        fig.colorbar(plt2, ax=ax2)

        ax3 = fig.add_subplot(4, 1, 3, projection=ccrs.PlateCarree())
        plt3 = ax3.contourf(lon, lat, bathy_grd, transform=ccrs.PlateCarree())
        ax3.set_title('bathymetry')
        fig.colorbar(plt3, ax=ax3)

        ax4 = fig.add_subplot(4, 1, 4, projection=ccrs.PlateCarree())
        plt4 = ax4.contourf(lon, lat, D, transform=ccrs.PlateCarree())
        ax4.set_title('distance to coast')
        fig.colorbar(plt4, ax=ax4)

        fig.savefig('cell_width.png', bbox_inches='tight', dpi=400)
        plt.close()

        return cell_width

    def limit_spacing_gradient(self, lon, lat, cell_width,
                               sphere_radius, dhdx):

        print("Smoothing h(x) via |dh/dx| limits...")

        opts = jigsawpy.jigsaw_jig_t()
        opts.hfun_file = "spac_pre_smooth.msh"
        opts.jcfg_file = "opts_pre_smooth.jig"
        opts.verbosity = +1

        spac = jigsawpy.jigsaw_msh_t()
        spac.mshID = "ellipsoid-grid"
        spac.radii = np.full(3, sphere_radius, dtype=spac.REALS_t)
        spac.xgrid = np.radians(lon)
        spac.ygrid = np.radians(lat)
        spac.value = cell_width.astype(spac.REALS_t)
        spac.slope = np.full(spac.value.shape, dhdx, dtype=spac.REALS_t)
        jigsawpy.savemsh(opts.hfun_file, spac)

        jigsawpy.cmd.marche(opts, spac)

        return spac.value

    def distance_to_shapefile_points(self, lon, lat,
                                     sphere_radius, reggrid=False):

        # Get coastline coordinates from shapefile
        features = cfeature.GSHHSFeature(scale='l', levels=[1, 5])

        pt_list = []
        for feature in features.geometries():

            if feature.length < 20:
                continue

            for coord in feature.exterior.coords:
                pt_list.append([coord[0], coord[1]])

        coast_pts = np.radians(np.array(pt_list))

        # Convert coastline points to x,y,z and create kd-tree
        npts = coast_pts.shape[0]
        coast_pts_xyz = np.zeros((npts, 3))
        x, y, z = self.lonlat2xyz(coast_pts[:, 0], coast_pts[:, 1],
                                  sphere_radius)
        coast_pts_xyz[:, 0] = x
        coast_pts_xyz[:, 1] = y
        coast_pts_xyz[:, 2] = z
        tree = spatial.KDTree(coast_pts_xyz)

        # Make sure longitude and latitude are in radians
        if np.amax(lon) < 2.1 * np.pi:
            lonr = lon
            latr = lat
        else:
            lonr = np.radians(lon)
            latr = np.radians(lat)

        # Convert  backgound grid coordinates to x,y,z
        # and put in a nx x 3 array for kd-tree query
        if reggrid:
            Lon, Lat = np.meshgrid(lonr, latr)
        else:
            Lon = lonr
            Lat = latr
        X, Y, Z = self.lonlat2xyz(Lon, Lat, sphere_radius)
        pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        # Find distances of background grid coordinates to the coast
        print("   Finding distance")
        start = timeit.default_timer()
        d, idx = tree.query(pts)
        end = timeit.default_timer()
        print("   Done")
        print("   " + str(end - start) + " seconds")

        D = np.reshape(d, Lon.shape)

        return D

    def create_initial_points(self, meshfile, lon, lat, hfunction,
                              sphere_radius, outfile):

        # Open MPAS mesh and get cell variables
        nc_file = Dataset(meshfile, 'r')
        lonCell = nc_file.variables['lonCell'][:]
        latCell = nc_file.variables['latCell'][:]
        nCells = lonCell.shape[0]

        # Transform 0,360 range to -180,180
        idx, = np.where(lonCell > np.pi)
        lonCell[idx] = lonCell[idx] - 2.0 * np.pi

        # Interpolate hfunction onto mesh cell centers
        hfun = interpolate.RegularGridInterpolator(
            (np.radians(lon), np.radians(lat)),
            hfunction.T)
        mesh_pts = np.vstack((lonCell, latCell)).T
        hfun_interp = hfun(mesh_pts)

        # Find cells in refined region of waves mesh
        max_res = np.amax(hfunction)
        idx, = np.where(hfun_interp < 0.5 * max_res)

        # Find boundary cells
        #   some meshes have non-zero values
        #   in the extra columns of cellsOnCell
        #   in this case, these must be zeroed out
        #   to correctly identify boundary cells
        nEdgesOnCell = nc_file.variables['nEdgesOnCell'][:]
        cellsOnCell = nc_file.variables['cellsOnCell'][:]
        nz = np.zeros(cellsOnCell.shape, dtype=bool)
        for i in range(nCells):
            nz[i, 0:nEdgesOnCell[i]] = True
        cellsOnCell[~nz] = 0
        nCellsOnCell = np.count_nonzero(cellsOnCell, axis=1)
        is_boundary_cell = np.equal(nCellsOnCell, nEdgesOnCell)
        idx_bnd, = np.where(is_boundary_cell == False)  # noqa: E712

        # Force inclusion of all boundary cells
        idx = np.union1d(idx, idx_bnd)

        # Get coordinates of points
        lon = lonCell[idx]
        lat = latCell[idx]

        lon = np.append(lon, 0.0)
        lat = np.append(lat, 0.5 * np.pi)

        npt = lon.size

        # Change to Cartesian coordinates
        x, y, z = self.lonlat2xyz(lon, lat, sphere_radius)

        # Get coordinates and ID into structured array
        #   (for use with np.savetxt)
        pt_list = []
        for i in range(npt):
            # ID of -1 specifies that node is fixed
            pt_list.append((x[i], y[i], z[i], -1))
        pt_type = np.dtype({'names': ['x', 'y', 'z', 'id'],
                           'formats': [np.float64, np.float64,
                                       np.float64, np.int32]})
        pts = np.array(pt_list, dtype=pt_type)

        # Write initial conditions file
        f = open(outfile, 'w')
        f.write('# Initial coordinates \n')
        f.write('MSHID=3;EUCLIDEAN-MESH\n')
        f.write('NDIMS=3\n')
        f.write(f'POINT={npt}\n')
        np.savetxt(f, pts, fmt='%.12e;%.12e;%.12e;%2i')
        f.close()

        init = jigsawpy.jigsaw_msh_t()
        jigsawpy.loadmsh(self.opts.init_file, init)
        jigsawpy.savevtk("init.vtk", init)

        return

    def lonlat2xyz(self, lon, lat, R):

        x = R * np.multiply(np.cos(lon), np.cos(lat))
        y = R * np.multiply(np.sin(lon), np.cos(lat))
        z = R * np.sin(lat)

        return x, y, z
