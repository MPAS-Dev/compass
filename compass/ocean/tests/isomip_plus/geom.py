import numpy
from mpas_tools.mesh.interpolation import interp_bilin
import xarray


def define_thin_film_mask_step1(ds_mesh, ds_geom):
    """
    Define an MPAS mesh mask for the ocean domain including cells over the 
    full x- and y-range in order to include all land cells in the ocean's 
    thin-film region.

    Parameters
    ----------
    ds_mesh : xarray.Dataset
        An MPAS-Ocean mesh

    ds_geom : xarray.Dataset
        Ice-sheet topography produced by
        :py:class:`compass.ocean.tests.isomip_plus.process_geom.ProcessGeom`

    Returns
    -------
    ds_mask : xarray.Dataset
        A dataset containing ``regionCellMasks``, a field with the ocean mask
        that can be used to cull land cells from the mesh
    """
    x, y, x_cell, y_cell, ocean_fraction = _get_geom_fields(ds_geom, ds_mesh)

    ds_mask = xarray.Dataset()

    valid = numpy.logical_and(
        numpy.logical_and(x_cell >= x[0], x_cell <= x[-1]),
        numpy.logical_and(y_cell >= y[0], y_cell <= y[-1]))

    mask = valid

    n_cells = mask.shape[0]

    ds_mask['regionCellMasks'] = (('nCells', 'nRegions'),
                                  mask.astype(int).reshape(n_cells, 1))

    return ds_mask


def interpolate_ocean_mask(ds_mesh, ds_geom, min_ocean_fraction):
    """
    Interpolate the ocean mask from the original BISICLES grid to the MPAS
    mesh.  This is handled separately from other fields because the ocean mask
    is needed to cull land cells from the MPAS mesh before interpolating the
    remaining fields.

    Parameters
    ----------
    ds_mesh : xarray.Dataset
        An MPAS-Ocean mesh

    ds_geom : xarray.Dataset
        Ice-sheet topography produced by
        :py:class:`compass.ocean.tests.isomip_plus.process_geom.ProcessGeom`

    min_ocean_fraction : float
        The minimum ocean fraction after interpolation, below which the cell
        is masked as land (which is not distinguished from grounded ice)

    Returns
    -------
    ds_mask : xarray.Dataset
        A dataset containing ``regionCellMasks``, a field with the ocean mask
        that can be used to cull land cells from the mesh
    """
    x, y, x_cell, y_cell, ocean_fraction = _get_geom_fields(ds_geom, ds_mesh)

    ds_mask = xarray.Dataset()

    valid = numpy.logical_and(
        numpy.logical_and(x_cell >= x[0], x_cell <= x[-1]),
        numpy.logical_and(y_cell >= y[0], y_cell <= y[-1]))

    x_cell = x_cell[valid]
    y_cell = y_cell[valid]
    ocean_frac_observed = numpy.zeros(ds_mesh.sizes['nCells'])
    ocean_frac_observed[valid] = interp_bilin(x, y, ocean_fraction.values,
                                              x_cell, y_cell)

    mask = ocean_frac_observed > min_ocean_fraction

    n_cells = mask.shape[0]

    ds_mask['regionCellMasks'] = (('nCells', 'nRegions'),
                                  mask.astype(int).reshape(n_cells, 1))

    return ds_mask


def interpolate_geom(ds_mesh, ds_geom, min_ocean_fraction, thin_film_present):
    """
    Interpolate the ice geometry from the original BISICLES grid to the MPAS
    mesh.

    Parameters
    ----------
    ds_mesh : xarray.Dataset
        An MPAS-Ocean mesh

    ds_geom : xarray.Dataset
        Ice-sheet topography produced by
        :py:class:`compass.ocean.tests.isomip_plus.process_geom.ProcessGeom`

    min_ocean_fraction : float
        The minimum ocean fraction after interpolation, below which the cell
        is masked as land (which is not distinguished from grounded ice)

    thin_film_present: bool
        Whether domain contains a thin film below grounded ice

    Returns
    -------
    ds_out : xarray.Dataset
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
    x, y, x_cell, y_cell, ocean_fraction = _get_geom_fields(ds_geom, ds_mesh)

    ds_out = xarray.Dataset(ds_mesh)
    ds_out.attrs = ds_mesh.attrs

    ds_geom['oceanFraction'] = ocean_fraction

    # mash the topography to the ocean region before interpolation
    if not thin_film_present:
        for var in ['Z_bed', 'Z_ice_draft', 'floatingIceFraction',
                    'smoothedDraftMask']:
            ds_geom[var] = ds_geom[var] * ds_geom['oceanFraction']

    fields = {'bottomDepthObserved': 'Z_bed',
              'ssh': 'Z_ice_draft',
              'oceanFracObserved': 'oceanFraction',
              'landIceFraction': 'floatingIceFraction',
              'smoothedDraftMask': 'smoothedDraftMask'}

    valid = numpy.logical_and(
        numpy.logical_and(x_cell >= x[0], x_cell <= x[-1]),
        numpy.logical_and(y_cell >= y[0], y_cell <= y[-1]))

    if not numpy.all(valid) and not thin_film_present:
        raise ValueError('Something went wrong with culling.  There are still '
                         'out-of-range cells in the culled mesh.')

    for outfield, infield in fields.items():
        field = interp_bilin(x, y, ds_geom[infield].values, x_cell, y_cell)
        ds_out[outfield] = (('nCells',), field)

    ocean_frac_observed = ds_out['oceanFracObserved']
    if not numpy.all(ocean_frac_observed > min_ocean_fraction) and not thin_film_present:
        raise ValueError('Something went wrong with culling.  There are still '
                         'non-ocean cells in the culled mesh.')

    if not thin_film_present:
        for field in ['bottomDepthObserved', 'ssh', 'landIceFraction',
                      'smoothedDraftMask']:
            ds_out[field] = ds_out[field]/ocean_frac_observed

    return ds_out


def _get_geom_fields(ds_geom, ds_mesh):
    x = ds_geom.x.values
    y = ds_geom.y.values
    x_cell = ds_mesh.xCell.values
    y_cell = ds_mesh.yCell.values

    ocean_fraction = - ds_geom['landFraction'] + 1.0

    return x, y, x_cell, y_cell, ocean_fraction
