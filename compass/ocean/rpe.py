import xarray
import xarray.plot
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def compute_rpe(initial_state_file_name='initial_state.nc',
                output_file_prefix='output_', num_files=5):
    # --- Open and read vars from NC file 
    
    dsInit = xarray.open_dataset(initial_state_file_name)
    nCells = dsInit.sizes['nCells']
    nEdges = dsInit.sizes['nEdges']
    nVertLevels = dsInit.sizes['nVertLevels']

    xCell        = dsInit['xCell']
    yCell        = dsInit['yCell']
    xEdge        = dsInit['xEdge']
    yEdge        = dsInit['yEdge']
    areaCell     = dsInit['areaCell']
    maxLevelCell = dsInit['maxLevelCell']
    bottomDepth  = dsInit['bottomDepth']
    for n in range(num_files):
        # --- Write in text
        file1 = open('rpe_'+str(n)+'.txt','w')

        ds = xarray.open_dataset('{}{}.nc'.format(
                                 output_file_prefix, n+1))
        
        nt = ds.sizes['Time']
        xtime = ds['daysSinceStartOfSim']
        hFull = ds['layerThickness']
        densityFull = ds['density']
        
        ridgeDepth = 500.0
        bottomMax = np.max(bottomDepth)
        yMax = np.max(yEdge)
        yMin = np.min(yEdge)
        xMax = np.max(xEdge)
        
        gravity = 9.80616
        rpe = np.zeros(nt)
        vol_1D = np.zeros((nVertLevels*nCells))
        density_1D = np.zeros((nVertLevels*nCells))
        
        for t,ti in enumerate(xtime.values):
        
            print('t=',t) 
            h = hFull[ti,:,:].values
            density = densityFull[ti,:,:].values
        
            i = 0
            for iCell in range(nCells):
                for k in range(maxLevelCell[iCell].values):
                    vol_1D[i] = h[iCell,k]*areaCell[iCell]
                    density_1D[i] = density[iCell,k]
                    i = i+1
            nCells_1D = i
            # --- Density sorting in ascending order
            sorted_ind = np.argsort(density_1D)
            density_sorted = np.zeros(nCells_1D)
            vol_sorted = np.zeros(nCells_1D)
        
            density_sorted = density_1D[sorted_ind]
            vol_sorted = vol_1D[sorted_ind]
        
            for j in range(len(sorted_ind)):
                density_sorted[j] = density_1D[sorted_ind[j]]
                vol_sorted[j] = vol_1D[sorted_ind[j]]
        
            rpe1 = np.zeros(nCells_1D)
            sillWidth = np.zeros(nCells_1D)
            yWidth = np.zeros(nCells_1D)
            zMid = np.zeros(nCells_1D)
            z = 0.0
        
            # --- RPE computation
            for i in range(nCells_1D):
                yWidth[i] = yMax - yMin
                area = yWidth[i]*xMax
                thickness = vol_sorted[i]/area
                zMid[i] = z-thickness/2.0
                z = z-thickness
                rpe1[i] = gravity*density_sorted[i]*(zMid[i]+bottomMax)*vol_sorted[i]
        
            rpe[time] = np.sum(rpe1)/np.sum(areaCell)
        
            file1.write(str(t)+','+str(rpe[time])+"\n")
        
        file1.close()
        return
