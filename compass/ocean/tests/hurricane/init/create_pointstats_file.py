from compass.step import Step

import netCDF4
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt


class CreatePointstatsFile(Step):
    """
    A step for creating the input file for the pointwiseStats
    analysis member

    Attributes
    ----------
    mesh_file : str
       Name of mesh file

    station_files : str
        Files containing location of observation stations

    pointstats_file : str
        Name of output file contiaining pointstats information

    """
    def __init__(self, test_case, mesh, storm):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.hurricane.init.Init
            The test case this step belongs to

        mesh :  compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that creates the mesh used by this test case

        storm : str
            The name of the storm used
        """
        super().__init__(test_case=test_case, name='pointstats',
                         ntasks=1, min_tasks=1, openmp_threads=1)

        self.mesh_file = 'mesh.nc'
        if storm == 'sandy':
            self.station_files = ['NOAA-COOPS_stations.txt',
                                  'USGS_stations.txt']
        self.pointstats_file = 'points.nc'

        mesh_path = mesh.mesh_step.path

        self.add_input_file(
            filename=self.mesh_file,
            work_dir_target=f'{mesh_path}/culled_mesh.nc')

        for file in self.station_files:
            self.add_input_file(
                filename=f'{file}',
                target=f'{storm}_stations/{file}',
                database='hurricane')

        self.add_output_file(filename=self.pointstats_file)

    def create_pointstats_file(self, mesh_file, stations_files):
        """
        Find grid points nearest to observation stations
        and create pointwiseStats file
        """
        plt.switch_backend('agg')

        # Read in station locations
        lon = []
        lat = []
        for stations_file in stations_files:
            f = open(stations_file, 'r')
            lines = f.read().splitlines()
            for line in lines:
                lon.append(line.split()[0])
                lat.append(line.split()[1])

        # Convert station locations
        lon = np.radians(np.array(lon, dtype=np.float32))
        lon_idx, = np.where(lon < 0.0)
        lon[lon_idx] = lon[lon_idx] + 2.0*np.pi
        lat = np.radians(np.array(lat, dtype=np.float32))
        stations = np.vstack((lon, lat)).T

        # Read in cell center coordinates
        mesh_nc = netCDF4.Dataset(mesh_file, 'r')
        lonCell = np.array(mesh_nc.variables["lonCell"][:])
        latCell = np.array(mesh_nc.variables["latCell"][:])
        meshCells = np.vstack((lonCell, latCell)).T

        # Find nearest cell center to each station
        tree = spatial.KDTree(meshCells)
        d, idx = tree.query(stations)

        # Plot the station locations and nearest cell centers
        plt.figure()
        plt.plot(lonCell[idx], latCell[idx], '.')
        plt.plot(lon, lat, '.')
        plt.savefig('station_locations.png')

        # Open netCDF file for writing
        data_nc = netCDF4.Dataset(self.pointstats_file, 'w',
                                  format='NETCDF3_64BIT_OFFSET')

        # Find dimesions
        npts = idx.shape[0]
        ncells = lonCell.shape[0]

        # Declare dimensions
        data_nc.createDimension('nCells', ncells)
        data_nc.createDimension('StrLen', 64)
        data_nc.createDimension('nPoints', npts)

        # Declear variables
        npts = data_nc.dimensions['nPoints'].name
        pnt_ids = data_nc.createVariable('pointCellGlobalID',
                                         np.int32, (npts,))

        # Set variables
        pnt_ids[:] = idx[:] + 1
        data_nc.close()

    def run(self):
        """
        Run this step of the testcase
        """
        self.create_pointstats_file(self.mesh_file, self.station_files)
