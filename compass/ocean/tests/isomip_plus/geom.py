import numpy
from mpas_tools.mesh.interpolation import interp_bilin
import xarray


def define_thin_film_mask_step1(dsMesh, dsGeom):
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

    mask = valid

    nCells = mask.shape[0]

    dsMask['regionCellMasks'] = (('nCells', 'nRegions'),
                                 mask.astype(int).reshape(nCells, 1))

    return dsMask


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


def interpolate_geom(dsMesh, dsGeom, min_ocean_fraction, thin_film_present):
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

    thin_film_present: bool
        Whether domain contains a thin film below grounded ice

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
    if not thin_film_present:
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

    if not numpy.all(valid) and not thin_film_present:
        raise ValueError('Something went wrong with culling.  There are still '
                         'out-of-range cells in the culled mesh.')

    for outfield, infield in fields.items():
        field = interp_bilin(x, y, dsGeom[infield].values, xCell, yCell)
        dsOut[outfield] = (('nCells',), field)

    oceanFracObserved = dsOut['oceanFracObserved']
    if not numpy.all(oceanFracObserved > min_ocean_fraction) and not thin_film_present:
        raise ValueError('Something went wrong with culling.  There are still '
                         'non-ocean cells in the culled mesh.')

    if not thin_film_present:
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
