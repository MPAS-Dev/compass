import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import os 

%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (16, 16) # Large figures
dpi=200;

### axis_font = {'fontname':'Arial', 'size':'18'}    
title_font = {'fontname':'Arial', 'size':'32', 'color':'black', 'weight':'normal'}
matplotlib.rc('xtick', labelsize=28)
matplotlib.rc('ytick', labelsize=28)

def rmse(resTag, sliceTime):
# resTag is the resolution to compute RMSE
# sliceTag is the time slice, assumes 1 hour output and same for all cases
    fid = open(resTag+'/default/init_step/namelist.ocean','r')
    temp = fid.readlines()

    fid.close()
    for i, line in enumerate(temp):
        if "config_cosine_bell_lat_center" in line: 
            ii = line.find('=')+1
            latCent = float(line[ii:])
        if "config_cosine_bell_lon_center" in line: 
            ii = line.find('=')+1
            lonCent = float(line[ii:])
        if "config_cosine_bell_radius" in line: 
            ii = line.find('=')+1
            radius = float(line[ii:])
        if "config_cosine_bell_psi0" in line: 
            ii = line.find('=')+1
            psi0 = float(line[ii:])
        if "config_cosine_bell_vel_pd" in line: 
            ii = line.find('=')+1
            pd = float(line[ii:])

    init = xr.open_dataset('../../'+resTag+'/default/init_step/initial_state.nc')
    #find time since the beginning of run
    ds = xr.open_dataset('../../'+resTag+'/default/forward/output/output.0001-01-01_00.00.00.nc')
    tt=str(ds.xtime[sliceTime].values)
    tt.rfind('_')
    DY = float(tt[10:12])-1
    HR = float(tt[13:15])
    MN = float(tt[16:18])
    t = 86400.0*DY+HR*3600.+MN

    #find new location of blob center
    #center is based on equatorial velocity
    R = init.sphere_radius
    distTrav = 2.0*3.14159265*R / (86400.0*pd)*t
    #distance in radians is
    distRad = distTrav / R
    newLon = lonCent + distRad
    if newLon > 2.0*np.pi:
        newLon -= 2.0*np.pi

    #construct analytic tracer
    tracer = np.zeros_like(init.tracer1[0,:,0].values)
    latC = init.latCell.values
    lonC = init.lonCell.values
    for i in range(len(latC)):
        temp = R*np.arccos(np.sin(latCent)*np.sin(latC[i]) + np.cos(latCent)*np.cos(latC[i])*np.cos(lonC[i] - newLon))
        if temp < radius:
            tracer[i] = psi0 / 2.0 * (1.0 + np.cos(3.1415926*temp/radius))

    #oad forward mode data
    tracerF = ds.tracer1[sliceTime,:,0].values
    rmse = np.sqrt(np.mean((tracerF-tracer)**2))

    init.close()
    ds.close()
    return rmse, init.dims['nCells']

res = ['QU60','QU120','QU240']
xtemp = []
ytemp = []
for i in range(len(res)):
    exec('rmse'+res[i]+',nCells'+res[i]+' = rmse(res[i],24)')
    exec('xtemp.append(nCells'+res[i]+')')
    exec('ytemp.append(rmse'+res[i]+')')
xdata = np.asarray(xtemp)
ydata = np.asarray(ytemp)

p = np.polyfit(np.log10(xdata),np.log10(ydata),1)
conv = abs(p[0])*2.0

plt.loglog(xdata,ydata,'k')
plt.scatter(xdata,ydata,s=40,c='r')
plt.annotate('Order of Convergence = {}'.format(np.round(conv,2)),xy=((xdata[-1]+xdata[0])*0.4,ydata[0]),fontsize=28)
plt.savefig('convergence.png')

