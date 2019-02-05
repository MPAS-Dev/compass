#!/usr/bin/env python
import numpy
from netCDF4 import Dataset

from optparse import OptionParser

import os.path

import shapely.geometry

from progressbar import ProgressBar, Percentage, Bar, ETA

import glob


def computeSectionEdgeIndices(point1,point2):
    shapelySection = shapely.geometry.LineString([point1,point2])


    edgeIndices = []
    for iEdge in range(nEdges):
        if shapelySection.intersects(shapelyEdges[iEdge]):
            edgeIndices.append(iEdge)
    if(len(edgeIndices) == 0):
        return (numpy.array(edgeIndices),numpy.array([]))

    # favor larger x and y in a tie
    edgeIndices = numpy.array(edgeIndices)[numpy.argsort(xEdge[edgeIndices] + yEdge[edgeIndices])[::-1]]
    distanceSquared = (xEdge[edgeIndices] - point1[0])**2 + (yEdge[edgeIndices] - point1[1])**2
    currentEdge = edgeIndices[numpy.argmin(distanceSquared)]
    distanceSquared = (xEdge[edgeIndices] - point2[0])**2 + (yEdge[edgeIndices] - point2[1])**2
    endEdge = edgeIndices[numpy.argmin(distanceSquared)]
    xEnd = point2[0] #xEdge[endEdge]
    yEnd = point2[1] #yEdge[endEdge]


    # first vertex is the one closer to the end point
    vertices = verticesOnEdge[currentEdge,:]
    distances = (xVertex[vertices] - xEnd)**2 + (yVertex[vertices] - yEnd)**2
    iVertex = vertices[numpy.argmin(distances)]
  
    edgeIndices = [currentEdge]
    if(vertices[0] == iVertex):
        edgeSigns = [1]
    else:
        assert(vertices[1] == iVertex)
        edgeSigns = [-1]

    while(currentEdge != endEdge):
        edges = edgesOnVertex[iVertex,:]
        if(endEdge in edges):
            currentEdge = endEdge
        else:
            # favor larger x and y in a tie
            edges = edges[numpy.argsort(xEdge[edges] + yEdge[edges])[::-1]]
            minDistance = 1e30
            currentEdge = -1
            for iEdge in edges:
                if iEdge < 0:
                    continue
                if iEdge in edgeIndices:
                    continue
        
                distance = (xEdge[iEdge] - xEnd)**2 + (yEdge[iEdge] - yEnd)**2
                if(distance < minDistance):
                    minDistance = distance
                    currentEdge = iEdge

        if(currentEdge < 0):
            print "Warning: endEdge not found!"
            print xEdge[edgeIndices], yEdge[edgeIndices]
            print xEdge[endEdge], yEdge[endEdge]
            # we must be done done
            break
    
        edgeIndices.append(currentEdge)
        vertex1 = verticesOnEdge[currentEdge,0]
        vertex2 = verticesOnEdge[currentEdge,1]
        if(vertex1 == iVertex):
            iVertex = vertex2
            edgeSigns.append(1)
        else:
            assert(vertex2 == iVertex)
            iVertex = vertex1
            edgeSigns.append(-1)
  
    return (numpy.array(edgeIndices),numpy.array(edgeSigns))

                       
parser = OptionParser()
           
options, args = parser.parse_args()

if len(args) > 0:
    folder = args[0]
else:
    folder = '.'

inFileNames = sorted(glob.glob('%s/timeSeriesStatsMonthly.*.nc'%folder))

initFile = Dataset('%s/init.nc'%folder,'r')
outFileName = '%s/overturningStreamfunction.nc'%folder
continueOutput = os.path.exists(outFileName)
if(continueOutput):
    outFile = Dataset(outFileName,'r+',format='NETCDF4')
else:
    outFile = Dataset(outFileName,'w',format='NETCDF4')

nVertices = len(initFile.dimensions['nVertices'])
nCells = len(initFile.dimensions['nCells'])
nEdges = len(initFile.dimensions['nEdges'])
nVertLevels = len(initFile.dimensions['nVertLevels'])
nTimeIn = len(inFileNames)
if(continueOutput):
    nTimeOut = len(outFile.dimensions['nTime'])
else:
    nTimeOut = 0


verticesOnEdge = initFile.variables['verticesOnEdge'][:,:]-1 # change to zero-based indexing
cellsOnEdge = initFile.variables['cellsOnEdge'][:,:]-1 # change to zero-based indexing
dvEdge = initFile.variables['dvEdge'][:]
dcEdge = initFile.variables['dcEdge'][:]
cellsOnVertex = initFile.variables['cellsOnVertex'][:,:]-1 # change to zero-based indexing
nEdgesOnCell = initFile.variables['nEdgesOnCell'][:]
edgesOnCell = initFile.variables['edgesOnCell'][:,:]-1
edgesOnVertex = initFile.variables['edgesOnVertex'][:,:]-1
verticesOnCell = initFile.variables['verticesOnCell'][:,:]-1
maxLevelCell = initFile.variables['maxLevelCell'][:]-1
xEdge = initFile.variables['xEdge'][:]
yEdge = initFile.variables['yEdge'][:]
xVertex = initFile.variables['xVertex'][:]
yVertex = initFile.variables['yVertex'][:]
xCell = initFile.variables['xCell'][:]
yCell = initFile.variables['yCell'][:]


if(continueOutput):
    outOSF = outFile.variables['overturningStreamfunction']
    sectionIndicesArray = outFile.variables['sectionEdgeIndices'][:,:]
    sectionSignArray = outFile.variables['sectionEdgeSigns'][:,:]
    sectionEdgeIndices = []
    sectionEdgeSigns = []
    for xIndex in range(sectionIndicesArray.shape[0]):
        mask = sectionIndicesArray.mask[xIndex,:] == False
        sectionEdgeIndices.append(sectionIndicesArray[xIndex,:][mask])
        sectionEdgeSigns.append(sectionSignArray[xIndex,:][mask])
    x = outFile.variables['x'][:]
    z = outFile.variables['z'][:]
    nx = len(x)
    nz = len(z)
    
else:
    dx = 2e3 # 2 km
    yMin = numpy.amin(yVertex)
    yMax = numpy.amax(yVertex)
    xMin = 320e3 + 0.5*dx
    xMax = 800e3 - 0.5*dx
    nx = 240
    x = numpy.linspace(xMin, xMax, nx)
  
    dz = 5.0
    zMin = -720.0 + 0.5*dz
    zMax = 0.0 - 0.5*dz
    nz = 144
    z = numpy.linspace(zMax, zMin, nz)

    #print "Building shapely edges"
    shapelyEdges = []
    for iEdge in range(nEdges):
        points = []
        for v in verticesOnEdge[iEdge,:]:
            points.append((xVertex[v],yVertex[v]))
        shapelyEdges.append(shapely.geometry.LineString(points))

    #print "Finding section indices"
    maxSectionLength = 0
    sectionEdgeIndices = []
    sectionEdgeSigns = []
    pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(x)).start()
    for xIndex in range(len(x)):
        #print xIndex, '/', len(x)-1
        xSection = x[xIndex]
        (edgeIndices, edgeSigns) = computeSectionEdgeIndices(
            (xSection, yMin), (xSection, yMax))
        #if(len(edgeIndices) != 0):
        #    xMean = numpy.mean(xEdge[edgeIndices])
        #    x[xIndex] = xMean

        sectionEdgeIndices.append(edgeIndices)
        sectionEdgeSigns.append(edgeSigns)
        maxSectionLength = max(maxSectionLength,len(edgeIndices))
        pbar.update(xIndex+1)
    pbar.finish()
    
    sectionIndicesArray = numpy.ma.masked_all((nx,maxSectionLength),int)
    sectionSignsArray = numpy.ma.masked_all((nx,maxSectionLength),int)
    for xIndex in range(len(x)):
        count = len(sectionEdgeIndices[xIndex])
        sectionIndicesArray[xIndex,0:count] = sectionEdgeIndices[xIndex]
        sectionSignsArray[xIndex,0:count] = sectionEdgeSigns[xIndex]

    outFile.createDimension('nx', nx)
    outFile.createDimension('nz', nz)
    outFile.createDimension('maxSectionLength', maxSectionLength)
    outFile.createDimension('nTime', None)
    outOSF = outFile.createVariable('overturningStreamfunction', float,
                                    ['nTime', 'nz', 'nx'])
    outX = outFile.createVariable('x', float, ['nx'])
    outZ = outFile.createVariable('z', float, ['nz'])
    outX[:] = x
    outZ[:] = z
    outSectionIndices = outFile.createVariable('sectionEdgeIndices', int, 
                                               ['nx','maxSectionLength'])
    outSectionIndices[:, :] = sectionIndicesArray
    outSectionSigns = outFile.createVariable('sectionEdgeSigns', int,
                                             ['nx', 'maxSectionLength'])
    outSectionSigns[:, :] = sectionSignsArray

bottomDepth = initFile.variables['bottomDepth'][:]

#print "Building maxLevelEdgeTop"
maxLevelEdgeTop = -1*numpy.ones(nEdges,int)
for iEdge in range(nEdges):
    cell1 = cellsOnEdge[iEdge,0]
    cell2 = cellsOnEdge[iEdge,1]
    if(cell1 >= 0 and cell2 >= 0):
        maxLevelEdgeTop[iEdge] = min(maxLevelCell[cell1],maxLevelCell[cell2])

pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=nTimeIn*nx).start()
for tIndex in range(nTimeOut,nTimeIn):
    inFile = Dataset(inFileNames[tIndex], 'r')

    normalVelocity = \
        inFile.variables['timeMonthly_avg_normalVelocity'][0,:,:]
    layerThickness = \
        inFile.variables['timeMonthly_avg_layerThickness'][0,:,:]

    inFile.close()

    zInterfaceCell = numpy.zeros((nCells,nVertLevels+1))
    zInterfaceCell[:,-1] = -bottomDepth
    for levelIndex in range(nVertLevels-1,-1,-1):
        mask = levelIndex <= maxLevelCell
        zInterfaceCell[:,levelIndex] = zInterfaceCell[:,levelIndex+1] + mask*layerThickness[:,levelIndex]
  
        
    #print "Computing transport over sections"
    transportSection = numpy.zeros((nz-1,nx))
    transportSectionArea = numpy.zeros((nz-1,nx))
    for xIndex in range(nx):
        for sIndex in range(len(sectionEdgeIndices[xIndex])):
            iEdge = sectionEdgeIndices[xIndex][sIndex]
            sign = sectionEdgeSigns[xIndex][sIndex]

            cell1 = cellsOnEdge[iEdge,0]
            cell2 = cellsOnEdge[iEdge,1]
            levelIndex = maxLevelEdgeTop[iEdge]
            zBot = 0.5*(zInterfaceCell[cell1,levelIndex+1] + zInterfaceCell[cell2,levelIndex+1])
            layerThicknessEdge = 0.5*(layerThickness[cell1,levelIndex]
                              + layerThickness[cell2,levelIndex])
            zTop = zBot + layerThicknessEdge
            for zIndex in range(nz-2,-1,-1):
                #print zIndex
                # sum the fluxes (if any) within this level on the output grid
                while (levelIndex >= 0) and (zBot < z[zIndex]):
                    #print zIndex, levelIndex, z[zIndex], zBot
                    v = normalVelocity[iEdge,levelIndex]
                    if v == 0.0:
                        continue
                    if(zTop <= z[zIndex]):
                        dz = zTop - zBot
                        zBot = zTop
                        layerThicknessEdge = \
                            0.5*(layerThickness[cell1,levelIndex]
                                 + layerThickness[cell2,levelIndex])
                        zTop += layerThicknessEdge
                        levelIndex -= 1
                    else:
                        dz = z[zIndex] - zBot
                        zBot = z[zIndex]
                    area = dvEdge[iEdge]*dz
                    transportSection[zIndex,xIndex] += sign*area*v
                    transportSectionArea[zIndex,xIndex] += area
        pbar.update(tIndex*nx+xIndex+1)
    
    transportMask = 1.0*(transportSectionArea > 0.0)

    #print "Computing overturning streamfunction"
    # As transport betwee two points is the difference between OSF at these 
    # locations, OSF is the cumulative sum of transport
    osf = numpy.zeros((nz,nx))
    osf[1:,:] = 1e-6*numpy.cumsum(transportSection,axis=0) # in Sv
    # OSF is valid as long as there is valid transport on one side or the other
    osfMask = numpy.zeros((nz,nx))
    osfMask[0:-1,:] = transportMask
    osfMask[1:,:] = numpy.logical_or(osfMask[1:,:],transportMask)
  
    # make OSF a masked array for plotting
    osf = numpy.ma.masked_array(osf,osfMask == 0.0)

    outOSF[tIndex,:,:] = osf

pbar.finish()
outFile.close()

