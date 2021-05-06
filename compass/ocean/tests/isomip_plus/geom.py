import numpy
from netCDF4 import Dataset
import scipy.ndimage.filters as filters
from mpas_tools.mesh.interpolation import interp_bilin
import xarray


def process_input_geometry(inFileName, outFileName, filterSigma,
                           minIceThickness, scale=1.0):
    """
    Process the BISICLES input geometry from ISOMIP+.  This includes:

    * reading in the bathymetry and ice topography

    * scaling the ice draft

    * implementing a simple calving scheme based on a thickness threshold

    * smoothing the topography

    * writing the processed topography out to a file

    Parameters
    ----------
    inFileName : str
        The file name of the original BISICLES geometry

    outFileName : str
        The file name of the processed geometry

    filterSigma : float
        The distance over which to smooth the geometry (in km)

    minIceThickness : float
        The minimum ice thickness, below which it is calved (removed and
        replaced with open ocean)

    scale : float
        A fraction by which to scale the ice draft (as a very simple way of
        testing dynamic topography)
    """
    def readVar(varName, defaultValue=0.0):
        field = defaultValue * numpy.ones((ny, nx), float)
        field[buffer:-buffer, buffer:-buffer] = numpy.array(
            inFile.variables[varName])[:, minIndex:]
        return field

    def writeVar(outVarName, inVarName, field):
        outVar = outFile.createVariable(outVarName, 'f8', ('y', 'x'))
        inVar = inFile.variables[inVarName]
        outVar[:, :] = field
        outVar.setncatts({k: inVar.getncattr(k) for k in inVar.ncattrs()})

    buffer = 1

    x0 = 320e3  # km

    inFile = Dataset(inFileName, 'r')
    x = numpy.array(inFile.variables['x'])[:]
    y = numpy.array(inFile.variables['y'])[:]

    deltaX = x[1] - x[0]
    deltaY = y[1] - y[0]

    minIndex = numpy.nonzero(x >= x0)[0][0]

    nx = len(x) - minIndex + 2 * buffer
    ny = len(y) + 2 * buffer

    outX = x[minIndex] + deltaX * (-buffer + numpy.arange(nx))
    outY = y[0] + deltaY * (-buffer + numpy.arange(ny))

    surf = readVar('upperSurface')
    draft = readVar('lowerSurface')
    bed = readVar('bedrockTopography')
    floatingMask = readVar('floatingMask')
    groundedMask = readVar('groundedMask', defaultValue=1.0)
    openOceanMask = readVar('openOceanMask')

    iceThickness = surf - draft

    draft *= scale

    # take care of calving criterion
    mask = numpy.logical_and(floatingMask > 0.1,
                             iceThickness < minIceThickness)
    surf[mask] = 0.
    draft[mask] = 0.
    floatingMask[mask] = 0.
    openOceanMask[mask] = 1. - groundedMask[mask]

    bed, draft, smoothedDraftMask = _smooth_geometry(
        groundedMask, floatingMask, bed, draft, filterSigma)

    outFile = Dataset(outFileName, 'w', format='NETCDF4')
    outFile.createDimension('x', nx)
    outFile.createDimension('y', ny)

    outVar = outFile.createVariable('x', 'f8', ('x'))
    inVar = inFile.variables['x']
    outVar[:] = outX
    outVar.setncatts({k: inVar.getncattr(k) for k in inVar.ncattrs()})
    outVar = outFile.createVariable('y', 'f8', ('y'))
    inVar = inFile.variables['y']
    outVar[:] = outY
    outVar.setncatts({k: inVar.getncattr(k) for k in inVar.ncattrs()})
    writeVar('Z_ice_surface', 'upperSurface', surf)
    writeVar('Z_ice_draft', 'lowerSurface', draft)
    writeVar('Z_bed', 'bedrockTopography', bed)
    writeVar('floatingIceFraction', 'floatingMask', floatingMask)
    writeVar('landFraction', 'groundedMask', groundedMask)
    writeVar('openOceanFraction', 'openOceanMask', openOceanMask)
    writeVar('smoothedDraftMask', 'openOceanMask', smoothedDraftMask)

    outFile.close()
    inFile.close()


def interpolate_ocean_mask(dsMesh, dsGeom, min_ocean_fraction):
    """
    Interpolate the ocean mask from the original BISICLES grid to the MPAS
    mesh.  This is handled separately from other fields because the ocean mask
    is needed to cull land cells from the MPAS mesh before interpolating the
    remaining fields.

    Parameters
    ----------
    dsMesh : xarray.Dataset
        An MPAS-Ocean mesh

    dsGeom : xarray.Dataset
        Ice-sheet topography produced by
        :py:func:`compass.ocean.tests.isomip_plus.geom.process_input_geometry()`

    min_ocean_fraction : float
        The minimum ocean fraction after interpolation, below which the cell
        is masked as land (which is not distinguished from grounded ice)

    Returns
    -------
    dsMask : xarray.Dataset
        A dataset containing ``regionCellMasks``, a field with the ocean mask
        that can be used to cull land cells from the mesh
    """
    x, y, xCell, yCell, oceanFraction = _get_geom_fields(dsGeom, dsMesh)

    dsMask = xarray.Dataset()

    valid = numpy.logical_and(
        numpy.logical_and(xCell >= x[0], xCell <= x[-1]),
        numpy.logical_and(yCell >= y[0], yCell <= y[-1]))

    xCell = xCell[valid]
    yCell = yCell[valid]
    oceanFracObserved = numpy.zeros(dsMesh.sizes['nCells'])
    oceanFracObserved[valid] = interp_bilin(x, y, oceanFraction.values,
                                            xCell, yCell)

    mask = oceanFracObserved > min_ocean_fraction

    nCells = mask.shape[0]

    dsMask['regionCellMasks'] = (('nCells', 'nRegions'),
                                 mask.astype(int).reshape(nCells, 1))

    return dsMask


def interpolate_geom(dsMesh, dsGeom, min_ocean_fraction):
    """
    Interpolate the ice geometry from the original BISICLES grid to the MPAS
    mesh.

    Parameters
    ----------
    dsMesh : xarray.Dataset
        An MPAS-Ocean mesh

    dsGeom : xarray.Dataset
        Ice-sheet topography produced by
        :py:func:`compass.ocean.tests.isomip_plus.geom.process_input_geometry()`

    min_ocean_fraction : float
        The minimum ocean fraction after interpolation, below which the cell
        is masked as land (which is not distinguished from grounded ice)

    Returns
    -------
    dsOut : xarray.Dataset
        A dataset containing :

        * ``bottomDepthObserved`` -- the bedrock elevation (positive up)

        * ``ssh`` -- the sea surface height

        * ``oceanFracObserved`` -- the fraction of the mesh cell that is ocean

        * ``landIceFraction`` -- the fraction of the mesh cell that is covered
          by an ice shelf

        * ``smoothedDraftMask`` -- a smoothed version of the floating mask that
          may be useful for determining where to alter the vertical coordinate
          to accommodate ice-shelf cavities
    """
    x, y, xCell, yCell, oceanFraction = _get_geom_fields(dsGeom, dsMesh)

    dsOut = xarray.Dataset(dsMesh)
    dsOut.attrs = dsMesh.attrs

    dsGeom['oceanFraction'] = oceanFraction

    # mash the topography to the ocean region before interpolation
    for var in ['Z_bed', 'Z_ice_draft', 'floatingIceFraction',
                'smoothedDraftMask']:
        dsGeom[var] = dsGeom[var] * dsGeom['oceanFraction']

    fields = {'bottomDepthObserved': 'Z_bed',
              'ssh': 'Z_ice_draft',
              'oceanFracObserved': 'oceanFraction',
              'landIceFraction': 'floatingIceFraction',
              'smoothedDraftMask': 'smoothedDraftMask'}

    valid = numpy.logical_and(
        numpy.logical_and(xCell >= x[0], xCell <= x[-1]),
        numpy.logical_and(yCell >= y[0], yCell <= y[-1]))

    if not numpy.all(valid):
        raise ValueError('Something went wrong with culling.  There are still '
                         'out-of-range cells in the culled mesh.')

    for outfield, infield in fields.items():
        field = interp_bilin(x, y, dsGeom[infield].values, xCell, yCell)
        dsOut[outfield] = (('nCells',), field)

    oceanFracObserved = dsOut['oceanFracObserved']
    if not numpy.all(oceanFracObserved > min_ocean_fraction):
        raise ValueError('Something went wrong with culling.  There are still '
                         'non-ocean cells in the culled mesh.')

    for field in ['bottomDepthObserved', 'ssh', 'landIceFraction',
                  'smoothedDraftMask']:
        dsOut[field] = dsOut[field]/oceanFracObserved

    return dsOut


def _get_geom_fields(dsGeom, dsMesh):
    x = dsGeom.x.values
    y = dsGeom.y.values
    xCell = dsMesh.xCell.values
    yCell = dsMesh.yCell.values

    oceanFraction = - dsGeom['landFraction'] + 1.0

    return x, y, xCell, yCell, oceanFraction


def _smooth_geometry(landFraction, floatingFraction, bed, draft, filterSigma):
    """
    Smoothing is performed using only the topography in the portion of the grid
    that is ocean. This prevents the kink in the ice draft across the grounding
    line or regions of bare bedrock from influencing the smoothed topography.
    (Parts of the Ross ice shelf near the Trans-Antarctic Mountains are
    particularly troublesome if topogrpahy is smoothed across the grounding
    line.)

    Unlike in POP, the calving front is smoothed as well because MPAS-O does
    not support a sheer calving face
    """

    # we won't normalize bed topography or ice draft where the mask is below
    # this threshold
    threshold = 0.01

    oceanFraction = 1. - landFraction
    smoothedMask = filters.gaussian_filter(oceanFraction, filterSigma,
                                           mode='constant', cval=0.)
    mask = smoothedMask > threshold

    draft = filters.gaussian_filter(draft * oceanFraction, filterSigma,
                                    mode='constant', cval=0.)
    draft[mask] /= smoothedMask[mask]
    bed = filters.gaussian_filter(bed * oceanFraction, filterSigma,
                                  mode='constant', cval=0.)
    bed[mask] /= smoothedMask[mask]

    smoothedDraftMask = filters.gaussian_filter(floatingFraction, filterSigma,
                                                mode='constant', cval=0.)
    smoothedDraftMask[mask] /= smoothedMask[mask]

    return bed, draft, smoothedDraftMask
