# Creates topographic_wave_drag.nc file containing all data
# needed for each type of wave drag scheme.

import netCDF4
import numpy as np
import xarray as xr
from scipy import interpolate
from scipy.spatial import cKDTree

from compass.step import Step


class CalculateWaveDrag(Step):
    """
    A step for calculating wave drag information

    Attributes
    ----------
    bathy_file : str
       File name for the blended RTopo/GEBCO pixel file

    mesh_file : str
       File name for the culled mesh from the mesh test case

    bouy_file : str
       File name for the WOA bouyancy data

    output_file : str
       File name for the output file containing wave drag information
    """

    def __init__(self, test_case, mesh):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.tides.init.Init
            The test case this step belongs to

         mesh : compass.ocean.tests.tides.mesh.Mesh
            The test case the produces the mesh for this case

        """

        super().__init__(test_case=test_case, name='wave_drag',
                         ntasks=1, min_tasks=1, openmp_threads=1)

        self.bathy_file = 'bathy.nc'
        self.mesh_file = 'mesh.nc'
        self.buoy_file = 'buoy.nc'

        self.add_input_file(
            filename=self.buoy_file,
            target='WOA2013_buoyancies.nc',
            database='initial_condition_database')

        mesh_path = mesh.steps['cull_mesh'].path
        self.add_input_file(
            filename=self.mesh_file,
            work_dir_target=f'{mesh_path}/culled_mesh.nc')

        bathy_path = mesh.steps['pixel'].path
        pixel_file = f'{bathy_path}/RTopo_2_0_4_GEBCO_v2023_30sec_pixel.nc'
        self.add_input_file(
            filename=self.bathy_file,
            work_dir_target=pixel_file)

        self.output_file = 'topographic_wave_drag.nc'
        self.add_output_file(filename=self.output_file)

    def interpolate_data_to_grid(self, grid_file, data_file, var):
        """
        Interpolate a variable from a data file to an the MPAS grid
        """

        # Open files
        data_nc = netCDF4.Dataset(data_file, 'r')
        grid_nc = netCDF4.Dataset(grid_file, 'r')

        # Get grid from data file
        lon_data = data_nc.variables['longitude'][:]
        lat_data = data_nc.variables['latitude'][:]
        nsnaps = 1

        # Get grid from grid file
        lon_grid = np.mod(grid_nc.variables['lonEdge'][:] + np.pi,
                          2.0 * np.pi) - np.pi
        lon_grid = lon_grid * 180.0 / np.pi
        lat_grid = grid_nc.variables['latEdge'][:] * 180.0 / np.pi

        grid_points = np.column_stack((lon_grid, lat_grid))
        nEdges = lon_grid.size
        interp_data = np.zeros((nsnaps, nEdges))

        # Interpolate timesnaps
        print(f'Interpolating {var}')

        # Get data to interpolate
        data = data_nc.variables[var][:]
        # bottom_index = data_nc.variables['bottom_index'][:]
        data[data == np.nan] = 0
        data[data > 0.015] = 0.015
        data[data < 1.405189e-4] = 1.405189e-4
        # Interpolate data onto new grid
        interpolator = interpolate.RegularGridInterpolator(
            (lon_data, lat_data), data,
            bounds_error=False, fill_value=None, method='nearest')
        interp_data[0, :] = interpolator(grid_points)

        return interp_data

    def run(self):
        """
        Run the step to calculate wave drag information
        """

        # Open Datasets
        print("Opening Datasets")
        bd = xr.open_dataset(self.bathy_file)
        elev = np.asarray(bd.bed_elevation)  # (lat, lon)
        ylat = np.asarray(bd.lat.values[:], dtype=np.float64)
        xlon = np.asarray(bd.lon.values[:], dtype=np.float64)
        bed_slope = np.asarray(bd.bed_slope.values)
        xgrad = np.asarray(bd.bed_dz_dx.values)
        ygrad = np.asarray(bd.bed_dz_dy.values)

        md = xr.open_dataset(self.mesh_file)
        nEdges = np.size(md.nEdges.values)
        xEdges = np.asarray(md.xEdge.values)
        yEdges = np.asarray(md.yEdge.values)
        zEdges = np.asarray(md.zEdge.values)
        xyzVertex = np.vstack((xEdges, yEdges, zEdges)).T

        print("Getting nearest neighbors")
        tree = cKDTree(xyzVertex)
        xmid = .5 * (xlon[:-1:] + xlon[1::])
        ymid = .5 * (ylat[:-1:] + ylat[1::])
        indx = np.asarray(np.round(
            np.linspace(-1, ymid.size, 17)), dtype=int)
        nset = []

        for tile in range(indx.size - 1):

            head = indx[tile + 0] + 1
            tail = indx[tile + 1] + 1

            qpos = self.map_to_r3(xmid, ymid, head, tail)

            __, nloc = tree.query(qpos, workers=-1)

            nset.append(nloc)

        near = np.concatenate(nset)
        near = near.reshape((np.size(ymid), np.size(xmid)))
        del tree
        del qpos
        del nset
        del nloc

        print("Calculating Standard Deviation and Interpolating Bed Slope")
        stddev = np.zeros((nEdges))
        bed_slope_edges = np.zeros((nEdges))
        xGradEdges = np.zeros((nEdges))
        yGradEdges = np.zeros((nEdges))
        N1V_interp = np.zeros((nEdges))
        N1B_interp = np.zeros((nEdges))
        xlon_nn = [[] for _ in range(nEdges)]
        ylat_nn = [[] for _ in range(nEdges)]
        print("Making lists")
        for i in range(np.size(ymid)):
            for j in range(np.size(xmid)):
                ylat_nn, xlon_nn = self.make_nn_lists(ylat_nn, xlon_nn,
                                                      near, i, j)

        print("Calculating stats")
        for edge in range(nEdges):
            bed_slope_edge, sd, xGradEdge, yGradEdge = self.calc_stats(
                xlon_nn, ylat_nn, near, bed_slope,
                edge, xmid, ymid, elev, xgrad, ygrad)

            bed_slope_edges[edge] = bed_slope_edge
            stddev[edge] = sd
            xGradEdges[edge] = xGradEdge
            yGradEdges[edge] = yGradEdge

        # Fix some nn issues
        print("Fixing nans in data")
        i = 0
        while ((np.sum(np.isnan(stddev)) > 0) and (i < 10)):
            print("stddev", np.sum(np.isnan(stddev)))
            stddev = self.fix_nans(stddev, md)
            i = i + 1
        i = 0
        while ((np.sum(np.isnan(bed_slope_edges)) > 0) and (i < 10)):
            print("bed_slope_edges", np.sum(np.isnan(stddev)))
            bed_slope_edges = self.fix_nans(bed_slope_edges, md)
            i = i + 1
        i = 0
        while ((np.sum(np.isnan(xGradEdges)) > 0) and (i < 10)):
            print("xGradEdges", np.sum(np.isnan(stddev)))
            xGradEdges = self.fix_nans(xGradEdges, md)
            i = i + 1
        i = 0
        while ((np.sum(np.isnan(yGradEdges)) > 0) and (i < 10)):
            print("yGradEdges", np.sum(np.isnan(stddev)))
            yGradEdge = self.fix_nans(yGradEdges, md)
            i = i + 1

        # Remove any 0s from the bottom Depth
        # bottomDepthEdges[bottomDepthEdges==0.0] = 0.01

        N1V_interp = self.interpolate_data_to_grid(self.mesh_file,
                                                   self.buoy_file, 'N1V')
        N1B_interp = self.interpolate_data_to_grid(self.mesh_file,
                                                   self.buoy_file, 'N1B')
        twd = xr.Dataset()
        twd["topo_buoyancy_N1V"] = xr.DataArray(N1V_interp[0, :],
                                                dims={"nEdges"})
        twd["topo_buoyancy_N1B"] = xr.DataArray(N1B_interp[0, :],
                                                dims={"nEdges"})
        twd["bathy_stddev"] = xr.DataArray(stddev, dims={"nEdges"})
        twd["bed_slope_edges"] = xr.DataArray(bed_slope_edges, dims={"nEdges"})
        twd["lonGradEdge"] = xr.DataArray(xGradEdges, dims={"nEdges"})
        twd["latGradEdge"] = xr.DataArray(yGradEdges, dims={"nEdges"})
        twd.to_netcdf(self.output_file)

        del xlon_nn
        del ylat_nn

    def fix_nans(self, data, mesh_nc):
        """
        Replace NaN values with average of surrounding (non-NaN) values
        """
        if (np.sum(np.isnan(data))):
            edgesOnEdge = mesh_nc.edgesOnEdge.data

            nanEdges = np.isnan(data)
            for edge in range(mesh_nc.nEdges.data[-1]):
                if nanEdges[edge]:
                    surrounding = edgesOnEdge[edge]
                    surrounding = surrounding[surrounding > 0]
                    surrounding = surrounding[~np.isnan(data[surrounding])]

                    data[edge] = np.mean(data[surrounding])

        return data

    def make_nn_lists(self, ylat_nn, xlon_nn, near, i, j):
        """
        Create the nearest neighbor lists
        """
        ylat_nn[near[i, j]].append(i)
        xlon_nn[near[i, j]].append(j)
        return ylat_nn, xlon_nn

    def calc_stats(self, xlon_nn, ylat_nn, near, bed_slope,
                   edge, xlon, ylat, elev, xgrad, ygrad):
        """
        Calculate the mean bathymetry slopes/gradients and stadard deviation
        """

        # Prepare for calculating area integral
        # dlat = ylat[1] - ylat[0]
        # dlon = xlon[1] - xlon[0]
        # R = 6378206.4

        # Standard deviation
        xlon_idx = xlon_nn[edge]
        ylat_idx = ylat_nn[edge]
        Y = ylat[ylat_idx]
        X = xlon[xlon_idx]
        values = elev[ylat_idx, xlon_idx]

        # Fit: H = a+bx+cy+dxy
        coeffs = self.polyfit2d(X, Y, values)
        values = values - (coeffs[0] + coeffs[1] * X +
                           coeffs[2] * Y + coeffs[3] * Y * X)
        stddev = np.std(values)

        # Bed Slope
        nnlat = ylat_nn[edge]
        nnlon = xlon_nn[edge]

        # nndA = dlat * dlon * R * R * np.cos(nnlat)
        # dA = np.sum(nndA)

        nnbs = bed_slope[nnlat, nnlon]
        # nnel = elev[nnlat, nnlon]
        nnxg = xgrad[nnlat, nnlon]
        nnyg = ygrad[nnlat, nnlon]

        mean_slope = np.mean(nnbs)
        mean_xgrad = np.mean(nnxg)
        mean_ygrad = np.mean(nnyg)
        # mean_slope= np.sum(nnbs*nndA)/dA
        # mean_elev = np.sum(nnbs*nndA)/dA
        # mean_xgrad =np.sum(nnxg*nndA)/dA
        # mean_ygrad = np.sum(nnyg*nndA)/dA

        # return mean_slope, stddev, mean_elev, mean_xgrad, mean_ygrad
        return mean_slope, stddev, mean_xgrad, mean_ygrad

    def polyfit2d(self, X, Y, Z):
        """
        Calculate linear least squares fit for standard deviation calculation
        """
        # X, Y = np.meshgrid(x, y, copy=False)
        # X = X.flatten()
        # Y = Y.flatten()

        # Fit: H = a+bx+cy+dxy (from Jayne and St. Laurent)
        A = np.array([X * 0 + 1, X, Y, X * Y]).T
        # B = Z.flatten()
        B = Z

        coeffs, __, __, __ = np.linalg.lstsq(A, B, rcond=None)

        return coeffs

    def map_to_r3(self, xlon, ylat, head, tail):
        """
        Map lon-lat coordinates to XYZ points. Restricted to the
        panel LAT[HEAD:TAIL] to manage memory use.

        """

        sinx = np.sin(xlon * np.pi / 180.)
        cosx = np.cos(xlon * np.pi / 180.)
        siny = np.sin(ylat * np.pi / 180.)
        cosy = np.cos(ylat * np.pi / 180.)

        sinu, sinv = np.meshgrid(sinx, siny[head:tail])
        cosu, cosv = np.meshgrid(cosx, cosy[head:tail])

        rsph = 6378206.4

        xpos = rsph * cosu * cosv
        ypos = rsph * sinu * cosv
        zpos = rsph * sinv

        return np.vstack(
            (xpos.ravel(), ypos.ravel(), zpos.ravel())).T
