#!/usr/bin/env python
# Simple script to inject bathymetry onto a mesh
# Phillip Wolfram, 01/19/2018

import matplotlib.pyplot as plt
from open_msh import readmsh
import numpy as np
from scipy import interpolate
import netCDF4 as nc4
import timeit
import pprint
import os

dtor = np.pi/180.0
rtod = 180.0/np.pi

def interpolate_SRTM(lon_pts,lat_pts):
  
    # Open NetCDF data file and read cooordintes
    nc_data = nc4.Dataset("earth_relief_15s.nc","r")
    lon_data = nc_data.variables['lon'][:]*dtor
    lat_data = nc_data.variables['lat'][:]*dtor
    
    # Setup interpolation boxes (for large bathymetry datasets)
    n = 15  
    xbox = np.linspace(-180,180,n)*dtor
    ybox = np.linspace(-90,90,n)*dtor
    dx = xbox[1]-xbox[0]
    dy = ybox[1]-ybox[0]
    boxes = []
    for i in range(n-1):
      for j in range(n-1):
        boxes.append(np.asarray([xbox[i],xbox[i+1],ybox[j],ybox[j+1]]))

    # Initialize bathymetry
    bathymetry = np.zeros(np.shape(lon_mesh))
    bathymetry.fill(np.nan)

    # Interpolate inside each box
    start = timeit.default_timer()
    for i,box in enumerate(boxes):
      print i+1,"/",len(boxes)

      # Get data inside box (plus a small overlap region)
      overlap = 0.1
      lon_idx, = np.where((lon_data >= box[0]-overlap*dx) & (lon_data <= box[1]+overlap*dx))
      lat_idx, = np.where((lat_data >= box[2]-overlap*dy) & (lat_data <= box[3]+overlap*dy))
      xdata = lon_data[lon_idx]
      ydata = lat_data[lat_idx]
      zdata = nc_data.variables['z'][lat_idx,lon_idx]

      # Get points inside box
      lon_idx, = np.where((lon_pts >= box[0]) & (lon_pts <= box[1]))
      lat_idx, = np.where((lat_pts >= box[2]) & (lat_pts <= box[3]))
      idx = np.intersect1d(lon_idx,lat_idx)
      xpts = lon_mesh[idx]
      ypts = lat_mesh[idx]
      xy_pts = np.vstack((xpts,ypts)).T

      # Interpolate bathymetry onto points
      bathy = interpolate.RegularGridInterpolator((xdata,ydata),zdata.T,bounds_error=False,fill_value=np.nan)
      bathy_int = bathy(xy_pts)
      bathymetry[idx] = bathy_int
    
    end = timeit.default_timer()
    print end-start, " seconds"
 
    return bathymetry

def interpolate_topomsh(lon_pts,lat_pts):

    topo = readmsh('topo.msh')
    xpos = topo['COORD1']*dtor
    ypos = topo['COORD2']*dtor
    zlev = np.reshape(topo['VALUE'], (len(ypos), len(xpos)))

    Y, X = np.meshgrid(ypos, xpos)

    bathy = interpolate.LinearNDInterpolator(np.vstack((X.ravel(), Y.ravel())).T, zlev.ravel())
    bathymetry = bathy(np.vstack((lon_pts,lat_pts)).T)  

    return bathymetry

if __name__ == "__main__":
    import sys

    # Open NetCDF mesh file and read mesh points
    mesh_file = sys.argv[1]
    nc_mesh = nc4.Dataset(mesh_file,'r+')
    lon_mesh = np.mod(nc_mesh.variables['lonCell'][:] + np.pi, 2*np.pi)-np.pi
    lat_mesh = nc_mesh.variables['latCell'][:]

    # Interpolate bathymetry on to mesh points
    if os.path.isfile("earth_relief_15s.nc"):
      bathymetry = interpolate_SRTM(lon_mesh,lat_mesh)
    elif os.path.isfile("topo.msh"):
      bathymetry = interpolate_topomsh(lon_mesh,lat_mesh)
    else:
      print "Bathymetry data file not found"
      raise SystemExit(0)

    # Create new NetCDF variables in mesh file, if necessary
    nc_vars = nc_mesh.variables.keys()
    if 'bathymetry' not in nc_vars:
      nc_mesh.createVariable('bathymetry','f8',('nCells'))
    if 'cullCell' not in nc_vars: 
      nc_mesh.createVariable('cullCell','i',('nCells'))

    # Write to mesh file
    nc_mesh.variables['bathymetry'][:] = bathymetry 
    nc_mesh.variables['cullCell'][:] = nc_mesh.variables['bathymetry'][:] > 20.0
    nc_mesh.close()


