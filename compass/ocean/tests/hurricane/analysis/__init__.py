import datetime
import json
import os
from importlib import resources

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import xarray as xr
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from scipy import spatial

from compass.step import Step


class Analysis(Step):
    """
    A step for producing ssh validation plots at observation stations

    Attributes
    ----------
    frmt : str
        Format for datetimes

    min_date : str
        Beginning of time period to plot in frmt format

    max_data : str
        End of time period to plot in frmt format

    observation : dict
        Dictionary of stations belonging to a certain data product
    """
    def __init__(self, test_case, storm):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.hurricane.forward.Forward
            The test case this step belongs to

        storm : str
            The name of the storm to be plotted
        """

        super().__init__(test_case=test_case, name='analysis')

        self.add_input_file(filename='pointwiseStats.nc',
                            target='../forward/pointwiseStats.nc')
        self.add_input_file(filename='mesh.nc',
                            target='../forward/input.nc')

        self.frmt = '%Y %m %d %H %M'
        self.storm = storm

    def setup(self):
        """
        Setup test case and download data
        """
        package = self.__module__

        if self.storm == 'sandy':
            self.run_min_date = '2012 10 10 00 00'
            self.run_max_date = '2012 11 04 00 00'
            self.adjust_min_date = '2012 10 01 00 00'
            self.adjust_max_date = '2012 10 25 00 00'

            filename = 'sandy_stations.json'
            with resources.open_text(package, filename)as stations_file:
                self.observations = json.load(stations_file)

            for obs in self.observations:
                os.makedirs(f'{self.work_dir}/{obs}_data', exist_ok=True)
                self.add_input_file(
                    filename=f'{obs}_stations.txt',
                    target=f'sandy_stations/{obs}_stations.txt',
                    database='hurricane')
                for sta in self.observations[obs]:
                    self.add_input_file(
                        filename=f'{obs}_data/{sta}.txt',
                        target=f'sandy_validation/'
                               f'{obs}_stations/{sta}.txt',
                        database='hurricane')

            package = 'compass.ocean.tests.hurricane.init'
            filename = 'bathy_data.json'
            with resources.open_text(package, filename) as bathy_file:
                self.bathy_files = json.load(bathy_file)

            os.makedirs(f'{self.work_dir}/NCEI_data', exist_ok=True)
            os.makedirs(f'{self.work_dir}/LULC_data', exist_ok=True)
            for i, dem in enumerate(self.bathy_files["NCEI"]):
                self.add_input_file(
                    filename=f'NCEI_data/{dem}',
                    target=f'ncei/{dem}',
                    database='bathymetry_database')

                self.add_input_file(
                    filename=f'LULC_data/landuse_from_{dem}',
                    target=f'LULC/landuse_from_{dem}',
                    database='hurricane')

    def read_pointstats(self, pointstats_file):
        """
        Read the pointwiseStats data from the MPAS-Ocean run
        """
        pointstats_nc = netCDF4.Dataset(pointstats_file, 'r')
        data = {}
        data['date'] = pointstats_nc.variables['xtime'][:]
        data['datetime'] = []
        for date in data['date']:
            d = b''.join(date).strip()
            data['datetime'].append(
                datetime.datetime.strptime(
                    d.decode('ascii').strip('\x00'),
                    '%Y-%m-%d_%H:%M:%S'))
        data['datetime'] = np.asarray(data['datetime'], dtype='O')
        data['lon'] = np.degrees(
            pointstats_nc.variables['lonCellPointStats'][:])
        data['lon'] = np.mod(data['lon'] + 180.0, 360.0) - 180.0
        data['lat'] = np.degrees(
            pointstats_nc.variables['latCellPointStats'][:])
        data['ssh'] = pointstats_nc.variables['sshPointStats'][:]

        return data

    def read_station_data(self, obs_file, obs_type, min_date, max_date):
        """
        Read the observed ssh timeseries data for a given station
        """
        # Initialize variable for observation data
        obs_data = {}
        obs_data['ssh'] = []
        obs_data['datetime'] = []

        # Get data from observation file between min and max output times
        f = open(obs_file)
        obs = f.read().splitlines()
        for line in obs[1:]:
            if (line.find('#') >= 0 or
                    len(line.strip()) == 0 or not
                    line[0].isdigit()):
                continue
            if obs_type == 'NOAA-COOPS':
                # NOAA-COOPS format
                date = line[0:16]
                date_time = datetime.datetime.strptime(date, self.frmt)
                col = 5
                convert = 1.0
            elif obs_type == 'USGS':
                # USGS station format
                date = line[0:19]
                date_time = datetime.datetime.strptime(
                    date,
                    '%m-%d-%Y %H:%M:%S')
                col = 2
                convert = 0.3048
            min_datetime = datetime.datetime.strptime(min_date, self.frmt)
            max_datetime = datetime.datetime.strptime(max_date, self.frmt)
            if date_time >= min_datetime and date_time <= max_datetime:
                obs_data['datetime'].append(date_time)
                obs_data['ssh'].append(line.split()[col])

        # Convert observation data and replace fill values with nan
        obs_data['ssh'] = np.asarray(obs_data['ssh'])
        obs_data['ssh'] = obs_data['ssh'].astype(float) * convert
        fill_val = 99.0
        obs_data['ssh'][obs_data['ssh'] >= fill_val] = np.nan

        obs_data['datetime'] = np.asarray(obs_data['datetime'], dtype='O')

        return obs_data

    def read_station_file(self, station_file):
        """
        Read file containing station locations and names
        """
        stations = {}
        stations['name'] = []
        stations['lon'] = []
        stations['lat'] = []

        # Read in stations names and location
        f = open(station_file, 'r')
        lines = f.read().splitlines()
        for sta in lines:
            val = sta.split()
            stations['name'].append(val[2].strip("'"))
            stations['lon'].append(float(val[0]))
            stations['lat'].append(float(val[1]))
        stations['lon'] = np.asarray(stations['lon'])
        stations['lat'] = np.asarray(stations['lat'])

        return stations

    def adjust_station_data(self, obs_data):
        """
        Adjust mean sea level in observation data
        """
        adjust_max_date = datetime.datetime.strptime(self.adjust_max_date,
                                                     self.frmt)

        # Get mean sea level within adjust period
        val = 0.0
        cnt = 0.0
        for i in range(obs_data['datetime'].size):
            if obs_data['datetime'][i] < adjust_max_date:
                val = val + obs_data['ssh'][i]
                cnt = cnt + 1.0
        if cnt > 0.0:
            mean = val / cnt
        else:
            mean = 0.0

        # Correct observations for mean sea level
        obs_data['ssh'] = obs_data['ssh'] - mean

    def run(self):
        """
        Run this step of the test case
        """
        plt.switch_backend('agg')

        plot_station_dems = self.config.getboolean('hurricane_analysis',
                                                   'plot_station_dems')

        # Get paths to run data to plot
        pointstats_file = {}
        comparison_runs = self.config.get('hurricane_analysis',
                                          'analysis_runs')
        comparison_runs_sp = comparison_runs.split(',')
        if comparison_runs_sp[0] != '':
            for run in comparison_runs_sp:
                run_sp = run.split(':')
                run_name = run_sp[0]
                run_path = run_sp[1]
                run_file = f'{run_path}/pointwiseStats.nc'

                if os.path.isfile(run_file):
                    pointstats_file[run_name] = run_file
        else:
            print("No run paths specified for analysis")

        # Read in model point output data and create kd-tree
        data = {}
        tree = {}
        for run in pointstats_file:
            data[run] = self.read_pointstats(pointstats_file[run])
            points = np.vstack((data[run]['lon'], data[run]['lat'])).T
            tree[run] = spatial.KDTree(points)

        # Initialize for plotting high water marks after time series plots
        hwm_obs = []
        hwm_mod = {}
        station_lon = []
        station_lat = []
        for i, run in enumerate(data):
            hwm_mod[run] = []

        # Create new colormap for LULC
        tab20_colors = plt.cm.get_cmap('tab20').colors
        tab20 = []
        for c in tab20_colors:
            tab20.append(mcolors.to_hex(c))
        new_colors = [
            '#91a9b1',  # Seafoam green
            '#b46617',  # Burnt orange
        ]
        tab20.extend(new_colors)
        self.tab25 = mcolors.ListedColormap(tab20, name='tab25')

        # Plot time series
        for obs in self.observations:
            os.makedirs(f'{self.work_dir}/{obs}_plots', exist_ok=True)

            # Read in station file
            stations = self.read_station_file(f'{obs}_stations.txt')

            for sta in self.observations[obs]:

                print(sta)
                i = stations['name'].index(sta)

                # Read in observed data and get coordinates
                obs_data = self.read_station_data(f'{obs}_data/{sta}.txt', obs,
                                                  self.adjust_min_date,
                                                  self.run_max_date)

                self.adjust_station_data(obs_data)

                sta_lon = stations['lon'][i]
                sta_lat = stations['lat'][i]
                station_lon.append(sta_lon)
                station_lat.append(sta_lat)

                # Create figure
                fig = plt.figure(figsize=[6, 4])
                gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

                # Plot observation station location
                ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
                ax1.set_extent([sta_lon - 10.0, sta_lon + 10.00,
                               sta_lat - 7.0, sta_lat + 7.0],
                               crs=ccrs.PlateCarree())
                ax1.add_feature(cfeature.LAND, zorder=100)
                ax1.add_feature(cfeature.LAKES, alpha=0.5, zorder=101)
                ax1.coastlines('50m', zorder=101)
                ax1.plot(sta_lon, sta_lat, 'C0o', zorder=102)

                # Plot local observation station location
                ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
                ax2.set_extent([sta_lon - 2.5, sta_lon + 2.5,
                               sta_lat - 1.75, sta_lat + 1.75],
                               crs=ccrs.PlateCarree())
                ax2.add_feature(cfeature.LAND, zorder=100)
                ax2.add_feature(cfeature.LAKES, alpha=0.5, zorder=101)
                ax2.coastlines('50m', zorder=101)
                ax2.plot(sta_lon, sta_lat, 'C0o', zorder=102)

                # Plot observed data
                ax3 = fig.add_subplot(gs[1, :])
                l1, = ax3.plot(obs_data['datetime'], obs_data['ssh'], 'C0-')
                labels = ['Observed']
                lines = [l1]
                hwm_obs.append(np.max(obs_data['ssh']))

                for i, run in enumerate(data):

                    # Find closest output point to station location
                    d, idx = tree[run].query(np.asarray([sta_lon, sta_lat]))

                    # Plot output point location
                    ax1.plot(data[run]['lon'][idx],
                             data[run]['lat'][idx],
                             'C' + str(i + 1) + 'o')
                    ax2.plot(data[run]['lon'][idx],
                             data[run]['lat'][idx],
                             'C' + str(i + 1) + 'o')

                    # Plot modelled data
                    l2, = ax3.plot(data[run]['datetime'],
                                   data[run]['ssh'][:, idx],
                                   'C' + str(i + 1) + '-')
                    labels.append(run)
                    lines.append(l2)
                    hwm_mod[run].append(np.max(data[run]['ssh'][:, idx]))

                # Set figure labels and axis properties and save
                ax3.set_xlabel('time')
                ax3.set_ylabel('ssh (m)')
                plot_min_date = self.config.get('hurricane_analysis',
                                                'plot_min_date')
                plot_max_date = self.config.get('hurricane_analysis',
                                                'plot_max_date')
                min_date = datetime.datetime.strptime(plot_min_date, self.frmt)
                max_date = datetime.datetime.strptime(plot_max_date, self.frmt)
                ax3.set_xlim([min_date, max_date])
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                lgd = plt.legend(lines, labels, loc=9,
                                 bbox_to_anchor=(0.5, -0.5),
                                 ncol=3, fancybox=False, edgecolor='k')
                st = plt.suptitle('Station ' + sta, y=1.025, fontsize=16)
                fig.tight_layout()
                fig.savefig(f'{obs}_plots/{sta}.png', bbox_inches='tight',
                            bbox_extra_artists=(lgd, st,))
                plt.close()

                # Plot DEM and LULC around station
                if plot_station_dems:
                    self.plot_dem(sta, sta_lon, sta_lat)

        # Convert to numpy arrays
        hwm_obs = np.asarray(hwm_obs)
        for i, run in enumerate(data):
            hwm_mod[run] = np.asarray(hwm_mod[run])

        # Plot modeled vs. observed hwm scatter
        fig = plt.figure()
        ax = fig.add_subplot(111)
        labels = []
        scatters = []
        for i, run in enumerate(data):
            sc = ax.scatter(hwm_obs, hwm_mod[run], alpha=0.5)
            scatters.append(sc)
            labels.append(run)
        ln, = ax.plot(hwm_obs, hwm_obs, 'k')
        scatters.append(ln)
        labels.append('perfect agreement')
        lgd = plt.legend(scatters, labels, loc=9,
                         bbox_to_anchor=(0.5, -0.1),
                         ncol=3, fancybox=False, edgecolor='k')
        ax.set_xlabel('observed')
        ax.set_ylabel('modeled')
        fig.tight_layout()
        fig.savefig('hwm_mod_obs.png', bbox_inches='tight',
                    bbox_extra_artists=(lgd,))
        plt.close()

        # Plot geographic hwm error
        station_lon = np.asarray(station_lon)
        station_lat = np.asarray(station_lat)
        for run in data:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

            diff = hwm_mod[run] - hwm_obs
            cm = ax.scatter(station_lon, station_lat, c=diff, cmap='PuOr',
                            zorder=102, vmax=3.0, vmin=-3.0, edgecolor='k')

            ax.set_extent([np.min(station_lon) - 0.1,
                           np.max(station_lon) + 0.1,
                           np.min(station_lat) - 0.1,
                           np.max(station_lat) + 0.1],
                          crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, zorder=100, color='gray')
            ax.add_feature(cfeature.OCEAN, zorder=100)
            ax.add_feature(cfeature.LAKES, alpha=0.5, zorder=101)
            ax.coastlines('50m', zorder=101)
            cb = fig.colorbar(cm, extend='both')
            cb.set_label('max high water error (m)')
            fig.tight_layout()
            fig.savefig(f'hwm_spatial_{run}.png', bbox_inches='tight')
            plt.close()

    def plot_dem(self, sta, sta_lon, sta_lat):

        dsMesh = xr.open_dataset('mesh.nc')

        # Find DEM tile containing station
        lat_name = 'lat'
        lon_name = 'lon'
        for dem in self.bathy_files['NCEI']:
            ds_topo = xr.open_dataset(f'NCEI_data/{dem}')
            lon = ds_topo.lon.values
            lat = ds_topo.lat.values
            da_topo = ds_topo.Band1
            if sta_lon > np.min(lon) and sta_lon < np.max(lon) and \
               sta_lat > np.min(lat) and sta_lat < np.max(lat):

                if lat_name in da_topo.dims:
                    lat = da_topo[lat_name]
                    if lat.ndim == 1 and (lat.diff(lat_name) < 0).any():
                        da_topo = da_topo.sortby(lat_name)

                if lon_name in da_topo.dims:
                    lon = da_topo[lon_name]
                    if lon.ndim == 1 and (lon.diff(lon_name) < 0).any():
                        da_topo = da_topo.sortby(lon_name)

                break

        ds_lulc = xr.open_dataset(f'LULC_data/landuse_from_{dem}')
        da_lulc = ds_lulc.Band1

        fig = plt.figure(figsize=[18, 12])
        ax = []
        ax.append(fig.add_subplot(2, 2, 1))
        ax.append(fig.add_subplot(2, 2, 2))
        ax.append(fig.add_subplot(2, 2, 3))
        ax.append(fig.add_subplot(2, 2, 4))

        # Plot DEM topo and LULC around station
        for i, eps in enumerate([0.1, 0.01]):
            bbox = np.array([sta_lon - eps, sta_lon + eps,
                             sta_lat - eps, sta_lat + eps])

            patches = self.compute_cell_patches(dsMesh, bbox)

            skip_topo = False
            try:
                da = da_topo.sel(lon=slice(bbox[0], bbox[1]),
                                 lat=slice(bbox[2], bbox[3]))
            except KeyError:
                print('topo slicing failed')
                skip_topo = True

            if not skip_topo:
                axi = 2 * i
                da.plot(ax=ax[axi],
                        cmap=cmocean.cm.topo,
                        cbar_kwargs={'label': 'topo'})
                ax[axi].plot(sta_lon, sta_lat,
                             marker='o',
                             markerfacecolor='k',
                             markeredgecolor='r',
                             zorder=102)
                ax[axi].axis('equal')
                ax[axi].autoscale(enable=False)
                ax[axi].add_collection(patches)
                ax[axi].set_xlabel('longitude')
                ax[axi].set_ylabel('latitude')

                tick_locations = range(2, 24)
                tick_labels = ['h.i. dev',
                               'm.i. dev',
                               'l.i. dev',
                               'open dev',
                               'cul land',
                               'pasture',
                               'grassland',
                               'dec forest',
                               'eve forest',
                               'mix forest',
                               'scrub',
                               'p.f. wetland',
                               'p.s. wetland',
                               'p.e. wetland',
                               'e.f. wetland',
                               'e.s. wetland',
                               'e.e. wetland',
                               'u.c. shore',
                               'bare land',
                               'open water',
                               'p.a. bed',
                               'e.a. bed']

                da = da_lulc.sel(lon=slice(bbox[0], bbox[1]),
                                 lat=slice(bbox[2], bbox[3]))
                formt = plt.FuncFormatter(
                    lambda x, p: tick_labels[tick_locations.index(x)])

                axi = 2 * i + 1
                da.plot(ax=ax[axi],
                        cmap=self.tab25,
                        vmin=2,
                        vmax=23,
                        cbar_kwargs={'label': 'LULC',
                                     'ticks': tick_locations,
                                     'format': formt})
                ax[axi].plot(sta_lon, sta_lat,
                             marker='o',
                             markerfacecolor='k',
                             markeredgecolor='r',
                             zorder=102)
                ax[axi].axis('equal')
                ax[axi].set_xlabel('longitude')
                ax[axi].set_ylabel('latitude')

        fig.tight_layout()
        fig.savefig(f'{sta}_dem.png', bbox_inches='tight')

        plt.close()

    def compute_cell_patches(self, dsMesh, bbox):
        patches = []
        nVerticesOnCell = dsMesh.nEdgesOnCell.values
        verticesOnCell = dsMesh.verticesOnCell.values - 1
        lonVertex = np.degrees(dsMesh.lonVertex.values)
        latVertex = np.degrees(dsMesh.latVertex.values)
        lonVertex = np.mod(lonVertex + 180.0, 360.0) - 180.0
        lonCell = np.degrees(dsMesh.lonCell.values)
        latCell = np.degrees(dsMesh.latCell.values)
        lonCell = np.mod(lonCell + 180.0, 360.0) - 180.0
        for iCell in range(dsMesh.sizes['nCells']):
            if lonCell[iCell] < bbox[0] - 1:
                continue
            if lonCell[iCell] > bbox[1] + 1:
                continue
            if latCell[iCell] < bbox[2] - 1:
                continue
            if latCell[iCell] > bbox[3] + 1:
                continue

            nVert = nVerticesOnCell[iCell]
            vertexIndices = verticesOnCell[iCell, :nVert]
            vertices = np.zeros((nVert, 2))
            vertices[:, 0] = lonVertex[vertexIndices]
            vertices[:, 1] = latVertex[vertexIndices]

            in_box = False
            if np.any(vertices[:, 0] > bbox[0]) and \
               np.any(vertices[:, 0] < bbox[1]) and \
               np.any(vertices[:, 1] > bbox[2]) and \
               np.any(vertices[:, 1] < bbox[3]):
                in_box = True
            if not in_box:
                continue

            polygon = Polygon(vertices, closed=True)
            patches.append(polygon)

        p = PatchCollection(patches, alpha=0.5,
                            facecolor='none', edgecolor='k')

        return p
