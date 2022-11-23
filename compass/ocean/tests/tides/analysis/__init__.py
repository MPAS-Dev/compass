from compass.step import Step

import netCDF4
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from mpas_tools.logging import check_call


class Analysis(Step):
    """
    A step for producing harmonic constituent errors and validation plots

    Attributes
    ----------
    harmonic_analysis_file : str
        File containing MPAS-O constitents

    grid_file : str
        Name of file containing MPAS-O mesh information

    constituents : list
        List of constituents to extract from TPXO database

    tpxo_version : str
        Version of TPXO to use in validation
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.tides.forward.Forward
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='analysis')

        self.harmonic_analysis_file = 'harmonicAnalysis.nc'
        self.grid_file = 'initial_state.nc'
        self.constituents = ['k1', 'k2', 'm2', 'n2', 'o1', 'p1', 'q1', 's2']

        self.add_input_file(
            filename=self.harmonic_analysis_file,
            target='../forward/analysis_members/harmonicAnalysis.nc')

        self.add_input_file(
            filename=self.grid_file,
            target='../forward/initial_state.nc')

    def setup(self):
        """
        Setup test case and download data
        """

        config = self.config
        self.tpxo_version = config.get('tides', 'tpxo_version')

        os.makedirs(f'{self.work_dir}/TPXO_data', exist_ok=True)
        if self.tpxo_version == 'TPXO9':
            for constituent in self.constituents:
                self.add_input_file(
                    filename=f'TPXO_data/h_{constituent}_tpxo',
                    target=f'TPXO9/h_{constituent}_tpxo9_atlas_30_v5',
                    database='tides')
                self.add_input_file(
                    filename=f'TPXO_data/u_{constituent}_tpxo',
                    target=f'TPXO9/u_{constituent}_tpxo9_atlas_30_v5',
                    database='tides')
            self.add_input_file(
                filename='TPXO_data/grid_tpxo',
                target='TPXO9/grid_tpxo9_atlas_30_v5',
                database='tides')
        elif self.tpxo_version == 'TPXO8':
            for constituent in self.constituents:
                self.add_input_file(
                    filename=f'TPXO_data/h_{constituent}_tpxo',
                    target=f'TPXO8/hf.{constituent}_tpxo8_atlas_30c_v1.out',
                    database='tides')
                self.add_input_file(
                    filename=f'TPXO_data/u_{constituent}_tpxo',
                    target=f'TPXO8/uv.{constituent}_tpxo8_atlas_30c_v1.out',
                    database='tides')
            self.add_input_file(
                filename='TPXO_data/grid_tpxo',
                target='TPXO8/grid_tpxo8_atlas_30_v1',
                database='tides')

    def write_coordinate_file(self, idx):
        """
        Write mesh coordinates for TPXO extraction
        """

        # Read in mesh
        grid_nc = netCDF4.Dataset(self.grid_file, 'r')
        lon_grid = np.degrees(grid_nc.variables['lonCell'][idx])
        lat_grid = np.degrees(grid_nc.variables['latCell'][idx])
        nCells = len(lon_grid)

        # Write coordinate file for OTPS2
        f = open('lat_lon', 'w')
        for i in range(nCells):
            f.write(str(lat_grid[i])+'  '+str(lon_grid[i])+'\n')
        f.close()

    def setup_otps2(self):
        """
        Write input files for TPXO extraction
        """

        for con in self.constituents:
            print(f'setup {con}')

            # Lines for the setup_con files
            lines = [{'inp': f'inputs/Model_atlas_{con}',
                      'comment': '! 1. tidal model control file'},
                     {'inp': 'lat_lon',
                      'comment': '! 2. latitude/longitude/<time> file'},
                     {'inp': 'z',
                      'comment': '! 3. z/U/V/u/v'},
                     {'inp':  con,
                      'comment': '! 4. tidal constituents to include'},
                     {'inp': 'AP',
                      'comment': '! 5. AP/RI'},
                     {'inp': 'oce',
                      'comment': '! 6. oce/geo'},
                     {'inp': '1',
                      'comment': '! 7. 1/0 correct for minor constituents'},
                     {'inp': f'outputs/{con}.out',
                      'comment': '! 8. output file (ASCII)'}]

            # Create directory for setup_con and Model_atlas_con files
            if not os.path.exists('inputs'):
                os.mkdir('inputs')

            # Write the setup_con file
            f = open('inputs/'+con+'_setup', 'w')
            for line in lines:
                spaces = 28 - len(line['inp'])
                f.write(line['inp'] + spaces*' ' + line['comment'] + '\n')
            f.close()

            # Write the Model_atlas_con file
            f = open(f'inputs/Model_atlas_{con}', 'w')
            f.write(f'TPXO_data/h_{con}_tpxo\n')
            f.write(f'TPXO_data/u_{con}_tpxo\n')
            f.write('TPXO_data/grid_tpxo')
            f.close()

            # Create directory for the con.out files
            if not os.path.exists('outputs'):
                os.mkdir('outputs')

    def run_otps2(self):
        """
        Perform TPXO extraction
        """

        # Run the executable
        for con in self.constituents:
            print('')
            print(f'run {con}')
            check_call(f'extract_HC < inputs/{con}_setup',
                       logger=self.logger, shell=True)

    def read_otps2_output(self, idx):
        """
        Read TPXO extraction output
        """

        start = idx[0]
        for con in self.constituents:

            f = open(f'outputs/{con}.out', 'r')
            lines = f.read().splitlines()
            for i, line in enumerate(lines[3:]):
                line_sp = line.split()
                if line_sp[2] != '*************':
                    val = float(line_sp[2])
                    self.mesh_AP[con]['amp'][start+i] = val
                else:
                    self.mesh_AP[con]['amp'][start+i] = -9999

                if line_sp[3] != 'Site':
                    val = float(line_sp[3])
                    if val < 0:
                        val = val + 360.0
                    self.mesh_AP[con]['phase'][start+i] = val

                else:
                    self.mesh_AP[con]['phase'][start+i] = -9999

    def append_tpxo_data(self):
        """
        Inject TPXO data into harmonic analysis file
        """

        data_nc = netCDF4.Dataset(self.harmonic_analysis_file, 'a',
                                  format='NETCDF3_64BIT_OFFSET')
        for con in self.constituents:

            # Inject amplitude
            amp_varname = f'{con.upper()}Amplitude{self.tpxo_version}'
            amp_var = data_nc.createVariable(
                amp_varname,
                np.float64,
                ('nCells'))
            amp_var.units = 'm'
            amp_var.long_name = f'Amplitude of {con.upper()} tidal ' \
                'consitiuent at each cell center from the ' \
                f'{self.tpxo_version} model'
            amp_var[:] = self.mesh_AP[con]['amp'][:]

            # Inject phase
            phase_varname = f'{con.upper()}Phase{self.tpxo_version}'
            phase_var = data_nc.createVariable(
                phase_varname,
                np.float64,
                ('nCells'))
            phase_var.units = 'deg'
            phase_var.long_name = f'Phase of {con.upper()} tidal ' \
                'consitiuent at each cell center from the ' \
                f'{self.tpxo_version} model'
            phase_var[:] = self.mesh_AP[con]['phase'][:]

        data_nc.close()

    def check_tpxo_data(self):
        """
        Check if TPXO data exists in harmonic analysis file
        """

        data_nc = netCDF4.Dataset(self.harmonic_analysis_file, 'r',
                                  format='NETCDF3_64BIT_OFFSET')
        self.nCells = len(data_nc.dimensions['nCells'])

        for con in self.constituents[:]:
            amp_var = f'{con.upper()}Amplitude{self.tpxo_version}'
            phase_var = f'{con.upper()}Phase{self.tpxo_version}'
            if (amp_var in data_nc.variables) \
                    and (phase_var in data_nc.variables):

                self.constituents.remove(con)
                print(f'{con} TPXO Constituent already exists '
                      f'in {self.harmonic_analysis_file}')

        data_nc.close()

    def plot(self):
        """
        Calculate errors and plot consitituents
        """

        plt.switch_backend('agg')
        cmap_reversed = cm.get_cmap('Spectral_r')

        # Initialize plotting variables
        TW = 2                         # Tick width
        TL = 2                         # Tick length
        TF = 8                         # Tick label size

        # Open data file
        data_file = self.harmonic_analysis_file
        data_nc = netCDF4.Dataset(data_file, 'r')

        lon_grid = np.mod(data_nc.variables['lonCell'][:] + np.pi,
                          2.0 * np.pi) - np.pi
        lon_grid = lon_grid*180.0/np.pi
        lat_grid = data_nc.variables['latCell'][:]*180.0/np.pi

        nCells = lon_grid.size
        data1 = np.zeros((nCells))
        data2 = np.zeros((nCells))
        data1_phase = np.zeros((nCells))
        data2_phase = np.zeros((nCells))
        depth = np.zeros((nCells))
        area = np.zeros((nCells))

        depth[:] = data_nc.variables['bottomDepth'][:]
        area[:] = data_nc.variables['areaCell'][:]

        constituent_list = ['K1', 'M2', 'N2', 'O1', 'S2']

        # Use these to fix up the plots
        subplot_ticks = [[np.linspace(0, 0.65, 10), np.linspace(0, 0.65, 10),
                          np.linspace(0, 0.13, 10), np.linspace(0, 0.13, 10)],
                         [np.linspace(0, 1.4,  10), np.linspace(0, 1.4,  10),
                          np.linspace(0, 0.22, 10), np.linspace(0, 0.25, 10)],
                         [np.linspace(0, 0.22, 10), np.linspace(0, 0.22, 10),
                          np.linspace(0, 0.05, 10), np.linspace(0, 0.05, 10)],
                         [np.linspace(0, 0.5,  10), np.linspace(0, 0.5,  10),
                          np.linspace(0, 0.08, 10), np.linspace(0, 0.08, 10)],
                         [np.linspace(0, 0.7,  10), np.linspace(0, 0.7,  10),
                          np.linspace(0, 0.5,  10), np.linspace(0, 0.5,  10)]]

        for i, con in enumerate(constituent_list):

            print('')
            print(f' ====== {con} Constituent ======')

            # Get data
            data1[:] = data_nc.variables[
                f'{con}Amplitude'][:]
            data1_phase[:] = data_nc.variables[
                f'{con}Phase'][:]
            data2[:] = data_nc.variables[
                f'{con}Amplitude{self.tpxo_version}'][:]
            data2_phase[:] = data_nc.variables[
                f'{con}Phase{self.tpxo_version}'][:]

            data1_phase = data1_phase*np.pi/180.0
            data2_phase = data2_phase*np.pi/180.0

            # Calculate RMSE values
            rmse_amp = 0.5*(data1 - data2)**2
            rmse_com = 0.5*(data2**2 + data1**2) \
                - data1*data2*np.cos(data2_phase - data1_phase)

            # Calculate mean (global) values
            idx = np.where((depth > 20)
                           & (rmse_com < 1000) & (rmse_amp < 1000))
            area_tot = np.sum(area[idx])
            glo_rmse_amp = np.sqrt(np.sum(rmse_amp[idx]*area[idx])/area_tot)
            glo_rmse_com = np.sqrt(np.sum(rmse_com[idx]*area[idx])/area_tot)
            print('Global RMSE (Amp) = ', glo_rmse_amp)
            print('Global RMSE (Com) = ', glo_rmse_com)

            # Calculate shallow RMSE (<=1000m)
            idx = np.where((depth > 20) & (depth < 1000)
                           & (np.abs(lat_grid) < 66)
                           & (rmse_com < 1000) & (rmse_amp < 1000))
            area_tot = np.sum(area[idx])
            shal_rmse_amp = np.sqrt(np.sum(rmse_amp[idx]*area[idx])/area_tot)
            shal_rmse_com = np.sqrt(np.sum(rmse_com[idx]*area[idx])/area_tot)
            print('Shallow RMSE (Amp) = ', shal_rmse_amp)
            print('Shallow RMSE (Com) = ', shal_rmse_com)

            # Calculate deep RMSE (>1000m)
            idx = np.where((depth >= 1000) & (np.abs(lat_grid) < 66)
                           & (rmse_com < 1000) & (rmse_amp < 1000))
            area_tot = np.sum(area[idx])
            deep_rmse_amp = np.sqrt(np.sum(rmse_amp[idx]*area[idx])/area_tot)
            deep_rmse_com = np.sqrt(np.sum(rmse_com[idx]*area[idx])/area_tot)
            print('Deep RMSE (Amp) = ', deep_rmse_amp)
            print('Deep RMSE (Com) = ', deep_rmse_com)

            rmse_amp = rmse_amp**0.5
            rmse_com = rmse_com**0.5

            # Plot data
            fig = plt.figure(figsize=(18, 12))
            subplot_title = [f'{con} Amplitude (simulation) [m]',
                             f'{con} Amplitude (TPXO8) [m]',
                             f'{con} RMSE (Amplitude) [m]',
                             f'{con} RMSE (Complex) [m]']

            for subplot in range(0, 4):
                ax = fig.add_subplot(2, 2, subplot+1,
                                     projection=ccrs.PlateCarree())
                ax.set_title(subplot_title[subplot], fontsize=20)
                levels = subplot_ticks[i][subplot][:]

                # MPAS amplitude and phase
                if subplot == 0:
                    cf = ax.tricontourf(lon_grid, lat_grid, data1,
                                        levels=levels,
                                        transform=ccrs.PlateCarree(),
                                        cmap=cmap_reversed)
                    ax.tricontour(lon_grid, lat_grid, data1_phase,
                                  levels=10, linewidths=0.5, colors='k')

                # TPXO amplitude and phase
                elif subplot == 1:
                    ix = np.logical_and(data2_phase >= 0, data2_phase < 360)
                    cf = ax.tricontourf(lon_grid, lat_grid, data2,
                                        levels=levels,
                                        transform=ccrs.PlateCarree(),
                                        cmap=cmap_reversed)
                    ax.tricontour(lon_grid[ix], lat_grid[ix], data2_phase[ix],
                                  levels=10, linewidths=0.5, colors='k')

                # Amplitude RMSE
                elif subplot == 2:
                    cf = ax.tricontourf(lon_grid, lat_grid, rmse_amp,
                                        levels=levels,
                                        transform=ccrs.PlateCarree(),
                                        cmap='OrRd')

                # Complex RMSE
                elif subplot == 3:
                    cf = ax.tricontourf(lon_grid, lat_grid, rmse_com,
                                        levels=levels,
                                        transform=ccrs.PlateCarree(),
                                        cmap='OrRd')

                ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.LAND, zorder=100)
                ax.add_feature(cfeature.LAKES, alpha=0.5, zorder=101)
                ax.add_feature(cfeature.COASTLINE, zorder=101)
                ax.tick_params(axis='both', which='major',
                               length=TL, width=TW, labelsize=TF)
                cbar = fig.colorbar(cf, ax=ax,
                                    ticks=levels.round(2), shrink=0.6)
                cbar.ax.tick_params(labelsize=16)

            fig.tight_layout()
            global_err = str(round(glo_rmse_com*100, 3))
            deep_err = str(round(deep_rmse_com*100, 3))
            shallow_err = str(round(shal_rmse_com*100, 3))
            fig.suptitle(f'Complex RMSE: Global = {global_err} cm; '
                         f'Deep = {deep_err} cm; '
                         f'Shallow = {shallow_err} cm',
                         fontsize=20)
            plt.savefig(f'{con}_plot.png')
            plt.close()

    def run(self):
        """
        Run this step of the test case
        """

        # Check if TPXO values aleady exist in harmonic_analysis.nc
        self.check_tpxo_data()

        # Setup input files for TPXO extraction
        self.setup_otps2()

        # Setup chunking for TPXO extraction with large meshes
        indices = np.arange(self.nCells)
        nchunks = np.ceil(self.nCells/200000)
        index_chunks = np.array_split(indices, nchunks)

        # Initialize data structure for TPXO values
        self.mesh_AP = {}
        for con in self.constituents:
            self.mesh_AP[con] = {'amp': np.zeros((self.nCells)),
                                 'phase': np.zeros((self.nCells))}

        # Extract TPXO values
        for idx in index_chunks:
            self.write_coordinate_file(idx)
            self.run_otps2()
            self.read_otps2_output(idx)

        # Inject TPXO values
        self.append_tpxo_data()

        # Calulate and plot global errors
        self.plot()
