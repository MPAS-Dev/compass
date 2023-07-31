import datetime
import json
import os
from importlib import resources

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
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

    pointstats_file : dict
        Dictionary of pointwiseStats outputs to plot. Dictionary key
        becomes the lable in the legend.

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

        self.frmt = '%Y %m %d %H %M'
        self.storm = storm

    def setup(self):
        """
        Setup test case and download data
        """
        package = self.__module__

        if self.storm == 'sandy':
            self.min_date = '2012 10 24 00 00'
            self.max_date = '2012 11 04 00 00'
            self.pointstats_file = {'MPAS-O': './pointwiseStats.nc'}

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

    def read_pointstats(self, pointstats_file):
        """
        Read the pointwiseStats data from the MPAS-Ocean run
        """
        pointstats_nc = netCDF4.Dataset('pointwiseStats.nc', 'r')
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

    def run(self):
        """
        Run this step of the test case
        """
        plt.switch_backend('agg')

        # Read in model point output data and create kd-tree
        data = {}
        tree = {}

        for run in self.pointstats_file:
            data[run] = self.read_pointstats(self.pointstats_file[run])
            points = np.vstack((data[run]['lon'], data[run]['lat'])).T
            tree[run] = spatial.KDTree(points)

        for obs in self.observations:
            os.makedirs(f'{self.work_dir}/{obs}_plots', exist_ok=True)

            # Read in station file
            stations = self.read_station_file(f'{obs}_stations.txt')

            for sta in self.observations[obs]:

                i = stations['name'].index(sta)

                # Read in observed data and get coordinates
                obs_data = self.read_station_data(f'{obs}_data/{sta}.txt', obs,
                                                  self.min_date, self.max_date)
                sta_lon = stations['lon'][i]
                sta_lat = stations['lat'][i]

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

                # Set figure labels and axis properties and save
                ax3.set_xlabel('time')
                ax3.set_ylabel('ssh (m)')
                min_date = datetime.datetime.strptime(self.min_date, self.frmt)
                max_date = datetime.datetime.strptime(self.max_date, self.frmt)
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
