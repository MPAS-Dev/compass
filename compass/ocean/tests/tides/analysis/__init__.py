from compass.step import Step

import netCDF4
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from importlib import resources
from scipy import spatial
import json
import os
import subprocess
import yaml


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
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.hurricane.forward.Forward
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='analysis')


########################################################################
########################################################################

def write_coordinate_file(grid_file):

  # Read in mesh
  grid_file = grid_file
  grid_nc = netCDF4.Dataset(grid_file,'r')
  lon_grid = np.degrees(grid_nc.variables['lonCell'][:])
  lat_grid = np.degrees(grid_nc.variables['latCell'][:])
  nCells = len(grid_nc.dimensions['nCells'])
  print(nCells)

  # Write coordinate file for OTPS2
  f = open('lat_lon','w')
  for i in range(nCells):
    f.write(str(lat_grid[i])+'  '+str(lon_grid[i])+'\n')
  f.close()

########################################################################
########################################################################

def setup_otps2(constituents,tpxo_data_path,tpxo_version):

  for con in constituents:
    print('setup '+con)
 
    # Lines for the setup_con files
    lines = [{'inp':'inputs/Model_atlas_'+con ,      'comment':'! 1. tidal model control file'},
             {'inp':'lat_lon' ,                      'comment':'! 2. latitude/longitude/<time> file'},
             {'inp':'z' ,                            'comment':'! 3. z/U/V/u/v'},
             {'inp': con ,                           'comment':'! 4. tidal constituents to include'},
             {'inp':'AP' ,                           'comment':'! 5. AP/RI'},
             {'inp':'oce' ,                          'comment':'! 6. oce/geo'},
             {'inp':'1' ,                            'comment':'! 7. 1/0 correct for minor constituents'},
             {'inp':'outputs/'+con+'.out' ,          'comment':'! 8. output file (ASCII)'}]
    
    # Create directory for setup_con and Model_atlas_con files
    if not os.path.exists('inputs'):
      os.mkdir('inputs') 
   
    # Write the setup_con file
    f = open('inputs/'+con+'_setup','w')
    for line in lines:
      spaces = 28 - len(line['inp'])
      f.write(line['inp'] + spaces*' ' + line['comment'] + '\n')
    f.close()

    # Write the Model_atlas_con file
    f = open('inputs/Model_atlas_'+con,'w')
    if tpxo_version == 'TPXO8':
      f.write('TPXO8/hf.'\
          +con+'_tpxo8_atlas_30c_v1.out\n')
      f.write('TPXO8/uv.'\
          +con+'_tpxo8_atlas_30c_v1.out\n')
      f.write('TPXO8/grid_tpxo8_atlas_30_v1')
    elif tpxo_version == 'TPXO9':
      f.write('TPXO9/h_'\
          +con+'_tpxo9_atlas_30_v5\n')
      f.write('TPXO9/u_'\
          +con+'_tpxo9_atlas_30_v5\n')
      f.write('TPXO9/grid_tpxo9_atlas_30_v5')
    f.close()

    # Link the TPXO data directory
    subprocess.call('ln -sf ' + tpxo_data_path, shell=True)

    # Create directory for the con.out files
    if not os.path.exists('outputs'):
      os.mkdir('outputs') 

########################################################################
########################################################################

def run_otps2(exe_path,constituents):

  # Make the executable if necessary 
  pwd = os.getcwd()
  os.chdir(exe_path)
  subprocess.call('make extract_HC',shell=True)
  os.chdir(pwd)

  # Run the executable 
  for con in constituents:
    print('run '+con)
    subprocess.call(exe_path+'/extract_HC < inputs/'+con+'_setup',shell=True)

########################################################################
########################################################################

def read_otps2_output(constituents):

  bou_AP = {}
  for con in constituents:
    bou_AP[con] = {'amp':[], 'phase':[]}

    f = open('outputs/'+con+'.out','r')
    lines = f.read().splitlines()
    for line in lines[3:]:
      line_sp = line.split()
      if line_sp[2] != '*************':
        val = float(line_sp[2])
        bou_AP[con]['amp'].append(val)
      else:
        bou_AP[con]['amp'].append('-9999')

      if line_sp[3] != 'Site':
        val = float(line_sp[3])
        if val < 0:
          val = val + 360.0
        bou_AP[con]['phase'].append(val)
        
      else:
        bou_AP[con]['phase'].append(-9999)

    #pprint.pprint(bou_AP)

  return bou_AP

########################################################################
########################################################################

def append_tpxo_data(output_file,constituents,mesh_AP,tpxo_version):

  data_nc = netCDF4.Dataset(output_file,'a', format='NETCDF3_64BIT_OFFSET')
  for con in constituents:
    amp_var = data_nc.createVariable(con.upper()+'Amplitude'+tpxo_version,np.float64,('nCells'))
    amp_var[:] = mesh_AP[con]['amp'][:]
    amp_var.units = 'm'
    amp_var.long_name = 'Amplitude of '+con.upper()+ ' tidal consitiuent at each cell center from '+tpxo_version+' model'
    phase_var = data_nc.createVariable(con.upper()+'Phase'+tpxo_version,np.float64,('nCells'))
    phase_var[:] = mesh_AP[con]['phase'][:]
    phase_var.units = 'deg'
    phase_var.long_name = 'Phase of '+con.upper()+ ' tidal consitiuent at each cell center from '+tpxo_version+' model'


########################################################################
########################################################################

if __name__ == '__main__':

  pwd = os.getcwd()
  inputfile = pwd+'/inject_TPXO8.config'
  f = open(inputfile)
  cfg = yaml.load(f,yaml.Loader)
  if cfg['tpxo_data_path'].find('TPXO8') > 0:
    tpxo_version = 'TPXO8'
  elif cfg['tpxo_data_path'].find('TPXO9') > 0:
    tpxo_version = 'TPXO9'

  write_coordinate_file(cfg['grid_file'])  
  setup_otps2(cfg['constituents'],cfg['tpxo_data_path'],tpxo_version)
  run_otps2(cfg['otps2_exe_path'],cfg['constituents'])
  mesh_AP = read_otps2_output(cfg['constituents'])
  append_tpxo_data(cfg['output_file'],cfg['constituents'],mesh_AP,tpxo_version)

#####
plt.switch_backend('agg')
cartopy.config['pre_existing_data_dir'] = \
    os.getenv('CARTOPY_DIR', cartopy.config.get('pre_existing_data_dir'))
cmap_reversed = matplotlib.cm.get_cmap('Spectral_r')
#####

def main(argv):
      
    # Setup for argument input
    resolution = ''
    saltype = ''
    
    # Get arguments -- optional (for specifying resolution and run type in title)
    try:
        opts, args = getopt.getopt(argv,"hr:s:c:",["resolution=","saltype==","constituent="])
    except getopt.GetoptError:
        print('plot_rmse_fixed.py -r <resolution> -s <saltype>')
    for opt, arg in opts:
        if opt in ('-h'):
            print('plot_rmse_fixed.py -r <resolution> -s <saltype>')
            sys.exit()
        elif opt in ("-r", "--resolution"):
            resolution = arg
            print('Resolution is ', resolution)
        elif opt in ("-s", "--saltype"):
            saltype = arg
            print('SAL type is ', saltype)

    # Initialize plotting variables
    LW   = 3                         # plot line width
    MS   = 1.5                       # plot symbol size 
    LF   = 30                        # label font size
    TW   = 2                         #Tick width
    TL   = 2                         #Tick length
    TF   = 8  
    i    = 0
    
    # Open data file
    data_file = 'analysis_members/harmonicAnalysis.nc'
    data_nc = netCDF4.Dataset(data_file,'r')
 
    lon_grid = np.mod(data_nc.variables['lonCell'][:] + np.pi, 2.0 * np.pi) - np.pi
    lon_grid = lon_grid*180.0/np.pi
    lat_grid = data_nc.variables['latCell'][:]*180.0/np.pi
    nCells = lon_grid.size
    data1 = np.zeros((nCells))
    data2 = np.zeros((nCells))
    depth = np.zeros((nCells))
    area = np.zeros((nCells))
    depth[:] = data_nc.variables['bottomDepth'][:]
    area[:] = data_nc.variables['areaCell'][:]
   
    data1_phase = np.zeros((nCells))
    data2_phase = np.zeros((nCells))
    cell = 0
    rmse_sum = 0
    count = 0
    shallow_rmse_amp = 0.0
    deep_rmse_amp = 0.0
    areaTotal = 0.0

    constituent_list = ['K1','M2','N2','O1','S2']
    constituent_num = 0
    subplot = 0

    # Use these to fix up the plots
    subplot_levels = [[np.linspace(0,0.65,16), np.linspace(0,0.65,16), np.linspace(0,0.13,16), np.linspace(0,0.13,16)], \
                      [np.linspace(0,1.4, 16),  np.linspace(0,1.4,16), np.linspace(0,0.22,16), np.linspace(0,0.5,16)], \
                      [np.linspace(0,0.22,16), np.linspace(0,0.22,16), np.linspace(0,0.05,16),np.linspace(0,0.025, 16)], \
                      [np.linspace(0,0.5, 16), np.linspace(0,0.5, 16), np.linspace(0,0.08,16),np.linspace(0,0.08,16)], \
                      [np.linspace(0,0.7, 16), np.linspace(0,0.7,16), np.linspace(0,0.5, 16), np.linspace(0,0.5,16)]]

    subplot_ticks = [[np.linspace(0,0.65, 10), np.linspace(0,0.65,10), np.linspace(0,0.13,10), np.linspace(0,0.13,10)], \
                      [np.linspace(0,1.4, 10),  np.linspace(0,1.4,10), np.linspace(0,0.22,10), np.linspace(0,0.25,10)], \
                      [np.linspace(0,0.22,10), np.linspace(0,0.22,10), np.linspace(0,0.05,10),np.linspace(0,0.05, 10)], \
                      [np.linspace(0,0.5, 10), np.linspace(0,0.5, 10), np.linspace(0,0.08,10),np.linspace(0,0.08,10)], \
                      [np.linspace(0,0.7, 10), np.linspace(0,0.7,10), np.linspace(0,0.5, 10), np.linspace(0,0.5,10)]]

    # Calculates in order: ['K1','M2','N2','O1','S2']
    for constituent_num in range(0,4):
        constituent = constituent_list[constituent_num]

        print(" ====== " + constituent + " Constituent ======")
        
        # Get data
        data1[:] = data_nc.variables[constituent+'Amplitude'][:]
        data1_phase[:] = data_nc.variables[constituent+'Phase'][:]*np.pi/180
        data2[:] = data_nc.variables[constituent+'AmplitudeTPXO8'][:]  
        data2_phase[:] = data_nc.variables[constituent+'PhaseTPXO8'][:]*np.pi/180

        # Calculate RMSE values
        rmse_amp = 0.5*(data1 - data2)**2
        rmse_com = 0.5*(data2**2 + data1**2) - data1*data2*np.cos(data2_phase - data1_phase)

        # Calculate mean (global) values
        rmse_amp_sum = 0
        rmse_com_sum = 0
        areaTotalAmp = 0
        areaTotalCom = 0
        for cell in range(0,nCells-1):
            if (depth[cell] > 20) and (rmse_com[cell] < 1000) and (rmse_amp[cell] < 1000):
                if (rmse_amp[cell] < 10000):
                    rmse_amp_sum += rmse_amp[cell]*area[cell]
                    areaTotalAmp += area[cell]
                if (rmse_com[cell] < 10000):
                    rmse_com_sum += rmse_com[cell]*area[cell]
                    areaTotalCom += area[cell]
        global_rmse_amp = np.sqrt(rmse_amp_sum / areaTotalAmp)
        global_rmse_com = np.sqrt(rmse_com_sum / areaTotalCom)
        print('Global RMSE (Amp) = ', global_rmse_amp)
        print('Global RMSE (Com) = ', global_rmse_com)

        # Calculate shallow RMSE (<=1000m)
        rmse_amp_sum = 0
        rmse_com_sum = 0
        areaTotalAmp = 0
        areaTotalCom = 0
        for cell in range(0,nCells-1):
            if (abs(lat_grid[cell]) < 66) and (depth[cell] < 1000) and (depth[cell] > 20):
                if (rmse_amp[cell] < 10000):
                    rmse_amp_sum += rmse_amp[cell]*area[cell]
                    areaTotalAmp += area[cell]
                if (rmse_com[cell] < 10000):
                    rmse_com_sum += rmse_com[cell]*area[cell]
                    areaTotalCom += area[cell]
        shallow_rmse_amp = np.sqrt(rmse_amp_sum / areaTotalAmp)
        shallow_rmse_com = np.sqrt(rmse_com_sum / areaTotalCom)
        print('Shallow RMSE (Amp) = ', shallow_rmse_amp)
        print('Shallow RMSE (Com) = ', shallow_rmse_com)

        # Calculate deep RMSE (>1000m)
        rmse_amp_sum = 0
        rmse_com_sum = 0
        areaTotal = 0
        for cell in range(0,nCells-1):
            if (abs(lat_grid[cell]) < 66) and (depth[cell] >= 1000):
                rmse_amp_sum += rmse_amp[cell]*area[cell]
                rmse_com_sum += rmse_com[cell]*area[cell]
                areaTotal += area[cell]
        deep_rmse_amp = np.sqrt(rmse_amp_sum / areaTotal)
        deep_rmse_com = np.sqrt(rmse_com_sum / areaTotal)
        print('Deep RMSE (Amp) = ', deep_rmse_amp)
        print('Deep RMSE (Com) = ', deep_rmse_com)

        rmse_amp = rmse_amp**0.5
        rmse_com = rmse_com**0.5

        # Plot data
        fig=plt.figure(figsize=(18,12))
        subplot_title = [constituent+' Amplitude (simulation) [m]', constituent+' Amplitude (TPXO8) [m]', \
                         constituent+' RMSE (Amplitude) [m]', constituent+' RMSE (Complex) [m]']

        # Setup the subplot
        for subplot in range(0,4) :
            ax = fig.add_subplot(2,2,subplot+1,projection = ccrs.PlateCarree())
            ax.set_title(subplot_title[subplot],fontsize=20)
            levels = subplot_ticks[constituent_num][subplot][:]
            if subplot == 0 :
                cf = ax.tricontourf(lon_grid,lat_grid,data1,levels=levels,
                    transform=ccrs.PlateCarree(),cmap=cmap_reversed)
                ax.tricontour(lon_grid,lat_grid,data1_phase,levels=10, linewidths=0.5, colors='k')
            elif subplot == 1 :
                iid = np.logical_and(data2_phase>=0, data2_phase < 360)
                cf = ax.tricontourf(lon_grid,lat_grid,data2,levels=levels,
                    transform=ccrs.PlateCarree(),cmap=cmap_reversed)
                ax.tricontour(lon_grid[iid],lat_grid[iid],data2_phase[iid],levels=10, linewidths=0.5, colors='k')
            elif subplot == 2 :
                cf = ax.tricontourf(lon_grid,lat_grid,rmse_amp,levels=levels,
                    transform=ccrs.PlateCarree(),cmap='OrRd')
            elif subplot == 3 :
                cf = ax.tricontourf(lon_grid,lat_grid,rmse_com,levels=levels,
                    transform=ccrs.PlateCarree(),cmap='OrRd')
            ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, zorder=100)
            ax.add_feature(cfeature.LAKES, alpha=0.5, zorder=101)
            ax.add_feature(cfeature.COASTLINE, zorder=101)
            ax.tick_params(axis='both', which='major', length=TL, width=TW, labelsize=TF)
            cbar = fig.colorbar(cf,ax=ax,ticks=levels.round(2),shrink=0.6)
            cbar.ax.tick_params(labelsize=16) 

        fig.tight_layout()
        fig.suptitle(resolution + ' with run type: ' + saltype + '\n' +
                        'Complex: Global Avg = ' + str(round(global_rmse_com*100,3)) + 'cm' + ';'\
                        ' Deep RMSE = ' + str(round(deep_rmse_com*100,3)) + 'cm' + \
                        '; Shallow RMSE = ' + str(round(shallow_rmse_com*100,3)) + 'cm', \
                        fontsize=20)
        plt.savefig(constituent+'_plot.png')
        plt.close()

