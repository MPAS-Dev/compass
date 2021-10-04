import numpy

def add_boundary_arrays(mesh, x_nonperiodic, y_nonperiodic):
  """
  Computes boundary arrays necessary for LES test cases
  Values are appended to a MPAS mesh dataset

  if the domain is doubly periodic, returns array of all zeros
  as arrays are meaningless in that case.

  Parameters
  ----------
  mesh : xarray dataset containing an MPAS mesh

  x_nonperiodic : logical if x direction is non periodic

  y_nonperiodic : logical if y direction is non periodic
  """

  nCells = mesh.dims['nCells']
  nEdges = mesh.dims['nEdges']
  neonc = mesh.nEdgesOnCell.values
  eonc = mesh.edgesOnCell.values - 1
  conc = mesh.cellsOnCell.values - 1
  nVertLevels = mesh.dims['nVertLevels']
  mlc = mesh.maxLevelCell.values - 1
  cone = mesh.cellsOnEdge.values - 1
  dv = mesh.dvEdge.values
  dc = mesh.dcEdge.values

  if (not x_nonperiodic) and (not y_nonperiodic):
    mesh['distanceToBoundary'] = (['nCells', 'nVertLevels'], numpy.zeros((nCells,nVertLevels)))
    mesh['boundaryNormalAngle'] = (['nCells', 'nVertLevels'], numpy.zeros((nCells,nVertLevels)))
    mesh['boundaryCellMask'] = (['nCells', 'nVertLevels'], numpy.zeros((nCells,nVertLevels)).astype(int))
    return mesh

  #surface first and save until later
  indVals = []
  angle = mesh.angleEdge.values
  xc = mesh.xEdge.values
  yc = mesh.yEdge.values
  xCell = mesh.xCell.values
  yCell = mesh.yCell.values
  aCell = numpy.zeros(nCells)
  bCell = numpy.zeros(nCells)
  xEdge = []
  yEdge = []
  for i in range(nCells):
    if conc[i,:].min() < 0:
      aEdge = 0
      count = 0
      for j in range(neonc[i]):
        if conc[i,j] < 0:
          indVals.append(i)
          xEdge.append(xc[eonc[i,j]])
          yEdge.append(yc[eonc[i,j]])
          #compute dx, dy
          #check angle edge
          #add pi if needed
          #check if over 2pi correct if needed
          dx = xc[eonc[i,j]] - xCell[i]
          dy = yc[eonc[i,j]] - yCell[i]
          if dx<=0 and dy<=0: # first quadrant
            if angle[eonc[i,j]] >= numpy.pi/2.:
              aEdge += angle[eonc[i,j]] - numpy.pi
              count += 1
            else:
              aEdge += angle[eonc[i,j]]
              count += 1
          elif dx<=0 and dy>=0: # fourth quadrant, but reference to 0 not 2pi
            if angle[eonc[i,j]] < 3.0*numpy.pi/2.0:
              aEdge += angle[eonc[i,j]] + numpy.pi
              count += 1
            else:
              aEdge += angle[eonc[i,j]]
              count += 1
          elif dx>=0 and dy>=0: #third quadrant
            if angle[eonc[i,j]] < numpy.pi/2.0: # not in correct place
              aEdge += angle[eonc[i,j]] + numpy.pi
              count += 1
            else:
              aEdge += angle[eonc[i,j]]
              count += 1
          else: #quadrant 2
            if angle[eonc[i,j]] > 3.0*numpy.pi/2.0: #wrong place
              aEdge += angle[eonc[i,j]] - numpy.pi
              count += 1
            else:
              aEdge += angle[eonc[i,j]]
              count += 1
      if count > 0:
        aCellTemp = aEdge / count
        if aCellTemp > numpy.pi:
          aCellTemp = 2.0*numpy.pi - aCellTemp
          aCell[i] = aCellTemp
          bCell[i] = 1
#with angle and index arrays, now can do distance

  dist = numpy.zeros((nCells,nVertLevels))
  angleCells = numpy.zeros((nCells,nVertLevels))
  boundaryCells = numpy.zeros((nCells,nVertLevels))
  for i in range(nCells):
    d = numpy.sqrt((xCell[i] - xEdge)**2 + (yCell[i] - yEdge)**2)
    ind = d.argmin()
    angleCells[i,0] = aCell[indVals[ind]]
    boundaryCells[i,0] = bCell[i]
    dist[i,0] = d.min()  

  #now to the rest
  for k in range(1,nVertLevels):
    inds = numpy.where(k==mlc)[0]
    edgeList = []
    cellList = []
    if len(inds) > 0:
      #First step is forming list of edges bordering maxLC
      for i in range(len(inds)):
        for j in range(neonc[inds[i]]):
          if conc[inds[i],j] not in inds:
            edgeList.append(eonc[inds[i],j])
      for i in range(len(edgeList)):
        c1 = cone[edgeList[i],0]
        c2 = cone[edgeList[i],1]
        if c1 in inds:
          if c2 not in cellList:
            cellList.append(c2)
        if c2 in inds:
          if c1 not in cellList:
            cellList.append(c1)

      for i in range(len(cellList)):
        aEdge = 0
        count = 0
        for j in range(neonc[cellList[i]]):
          if eonc[cellList[i],j] in edgeList:
            indVals.append(cellList[i])
            xEdge.append(xc[eonc[cellList[i],j]])
            yEdge.append(yc[eonc[cellList[i],j]])
            #compute dx, dy
            #check angle edge
            #add pi if needed
            #check if over 2pi correct if needed
            dx = xc[eonc[cellList[i],j]] - xCell[cellList[i]]
            dy = yc[eonc[cellList[i],j]] - yCell[cellList[i]]
            if dx<=0 and dy<=0: # first quadrant
              if angle[eonc[cellList[i],j]] >= numpy.pi/2.:
                aEdge += angle[eonc[cellList[i],j]] - numpy.pi
                count += 1
              else:
                aEdge += angle[eonc[cellList[i],j]]
                count += 1
            elif dx<=0 and dy>=0: # fourth quadrant, but reference to 0 not 2pi
              if angle[eonc[cellList[i],j]] < 3.0*numpy.pi/2.0:
                aEdge += angle[eonc[cellList[i],j]] + numpy.pi
                count += 1
              else:
                aEdge += angle[eonc[cellList[i],j]]
                count += 1
            elif dx>=0 and dy>=0: #third quadrant
              if angle[eonc[cellList[i],j]] < numpy.pi/2.0: # not in correct place
                aEdge += angle[eonc[cellList[i],j]] + numpy.pi
                count += 1
              else:
                aEdge += angle[eonc[cellList[i],j]]
                count += 1
            else: #quadrant 2
              if angle[eonc[cellList[i],j]] > 3.0*numpy.pi/2.0: #wrong place
                aEdge += angle[eonc[cellList[i],j]] - numpy.pi
                count += 1
              else:
                aEdge += angle[eonc[cellList[i],j]]
                count += 1
        if count > 0:
          aCellTemp = aEdge / count
          if aCellTemp > numpy.pi:
            aCellTemp = 2.0*numpy.pi - aCellTemp
            aCell[cellList[i]] = aCellTemp
            bCell[cellList[i]] = 1

        for ii in range(nCells):
          d = numpy.sqrt((xCell[ii] - xEdge)**2 + (yCell[ii] - yEdge)**2)
          ind = d.argmin()
          angleCells[ii,k] = aCell[indVals[ind]]
          boundaryCells[ii,k] = bCell[ii]
          dist[ii,k] = d.min()
    else:
      dist[:,k] = dist[:,0]
      angleCells[:,k] = angleCells[:,0]
      boundaryCells[:,k] = boundaryCells[:,0]
  mesh['distanceToBoundary'] = (['nCells', 'nVertLevels'], dist)
  mesh['boundaryNormalAngle'] = (['nCells', 'nVertLevels'], angleCells)
  mesh['boundaryCellMask'] = (['nCells', 'nVertLevels'], boundaryCells.astype(int))

  return mesh
