"""
Recompute MALI's melt from its diagnostic output.

Both melt forms are functions of the thermal forcing at the ice draft, which
MALI writes as ``ismip6shelfMelt_TFdraft``.  That means melt can be
recomputed in Python for any melt parameter and any basin correction
``dT_b`` without re-running MALI, which is what the ``dT_b`` fit needs and
what the verification step compares against.
"""

import dataclasses

import numpy as np
import xarray as xr

from compass.landice.tests.ismip7_calibration.quadratic import (
    MALI,
    angle_from_sin_slope,
    local_quadratic_melt,
    nonlocal_quadratic_melt,
)

#: bit of ``cellMask`` marking floating ice
FLOATING_MASK_BIT = 4

#: seconds in a year, matching MALI's ``scyr`` and its noleap calendar
SECONDS_PER_YEAR = 31536000.0


def read_run(filename):
    """
    Read the fields a melt diagnostic run writes.

    Parameters
    ----------
    filename : str
        An ``output_melt.nc`` written by a
        :py:class:`~compass.landice.tests.ismip7_calibration.ais.run_state.RunState`
        step

    Returns
    -------
    fields : dict
        ``melt`` (kg m-2 yr-1, positive for melting), ``tf_draft``,
        ``floating``, ``basin`` (MALI's 1-based numbering), ``delta_t`` and
        ``area``
    """  # noqa: E501
    with xr.open_dataset(filename) as ds:
        bmb = ds['floatingBasalMassBal'].isel(Time=0)
        cell_mask = ds['cellMask'].isel(Time=0)
        fields = dict(
            melt=(-bmb * SECONDS_PER_YEAR).compute(),
            tf_draft=ds['ismip6shelfMelt_TFdraft'].isel(Time=0).compute(),
            floating=((cell_mask & FLOATING_MASK_BIT) > 0).compute(),
            basin=ds['ismip6shelfMelt_basin'].compute(),
            delta_t=ds['ismip6shelfMelt_deltaT'].compute(),
            area=ds['areaCell'].compute())
    return fields


def initial_draft(mesh_file, constants=None):
    """
    Reconstruct the ice draft from the mesh file, before the timestep.

    **The diagnostic run evolves the geometry.**  Melt is applied over the
    single timestep and thins the ice by up to a metre or so, so the
    ``lowerSurface`` and ``thickness`` written to the output are *post*-step
    while ``TFdraft`` was computed *pre*-step.  Any check that pairs output
    geometry with output melt is therefore inconsistent by up to a metre of
    draft; the initial draft has to come from the mesh file instead.

    Parameters
    ----------
    mesh_file : str
        The MALI mesh file the run started from

    constants : compass.landice.tests.ismip7_calibration.quadratic.Constants, optional
        The densities to use; defaults to MALI's

    Returns
    -------
    draft : xarray.DataArray
        The ice draft, negative below sea level
    """  # noqa: E501
    if constants is None:
        constants = MALI
    with xr.open_dataset(mesh_file) as ds:
        thickness = ds['thickness']
        bed = ds['bedTopography']
        if 'Time' in thickness.dims:
            thickness = thickness.isel(Time=0)
        if 'Time' in bed.dims:
            bed = bed.isel(Time=0)
        thickness = thickness.compute()
        bed = bed.compute()

    floating_draft = -constants.rho_ice / constants.rho_ocean * thickness
    # where the ice is grounded the draft is the bed
    return xr.where(floating_draft > bed, floating_draft, bed)


def basin_mean_tf(tf_draft, area, floating, basin):
    """
    Area-weighted mean thermal forcing per basin, mapped back onto cells.

    This is what MALI's ISMIP6 non-local form computes internally, and it
    must be area-weighted: on a 4-20 km mesh a plain mean is wrong.

    Parameters
    ----------
    tf_draft : xarray.DataArray
        Thermal forcing at the ice draft

    area : xarray.DataArray
        Cell area

    floating : xarray.DataArray
        True on floating cells

    basin : xarray.DataArray
        Basin number per cell

    Returns
    -------
    mean_tf : xarray.DataArray
        The basin mean, broadcast back onto every cell
    """
    weights = area.where(floating)
    numer = (tf_draft * weights).groupby(basin.rename('basin')).sum()
    denom = weights.groupby(basin.rename('basin')).sum()
    means = numer / denom.where(denom != 0.0)
    return means.sel(basin=basin.rename('basin')).drop_vars('basin')


def melt_from_tf(melt_form, parameter, fields, config, delta_t=None,
                 mean_tf=None):
    """
    Recompute the melt field from the thermal forcing at the draft.

    Parameters
    ----------
    melt_form : {'ismip7', 'ismip6'}
        Which melt form to evaluate

    parameter : float or xarray.DataArray
        ``K`` for ``'ismip7'`` or ``gamma0`` for ``'ismip6'``

    fields : dict
        From :py:func:`read_run`

    config : compass.config.CompassConfigParser
        Configuration options, for the constant salinity, slope and Coriolis
        parameter

    delta_t : float or xarray.DataArray, optional
        The basin correction to apply; defaults to the field the run used

    mean_tf : xarray.DataArray, optional
        Precomputed basin-mean thermal forcing, for ``'ismip6'``

    Returns
    -------
    melt : xarray.DataArray
        Melt rate in kg m-2 yr-1, positive for melting, zero off the shelves
    """
    section = config['ismip7_calibration_melt']
    if delta_t is None:
        delta_t = fields['delta_t']

    tf_draft = fields['tf_draft']
    floating = fields['floating']

    if melt_form == 'ismip7':
        # config gives the *sine* of the slope while local_quadratic_melt
        # takes the angle
        slope = angle_from_sin_slope(section.getfloat('sin_slope'))
        constants = dataclasses.replace(
            MALI, coriolis=section.getfloat('coriolis'))
        melt = local_quadratic_melt(
            parameter, tf_draft, section.getfloat('salinity'), slope,
            constants=constants, delta_t=delta_t)
    elif melt_form == 'ismip6':
        if mean_tf is None:
            mean_tf = basin_mean_tf(tf_draft, fields['area'], floating,
                                    fields['basin'])
        melt = nonlocal_quadratic_melt(parameter, tf_draft, mean_tf,
                                       constants=MALI, delta_t=delta_t)
    else:
        raise ValueError(f"melt_form must be 'ismip7' or 'ismip6', but is "
                         f"'{melt_form}'")

    return xr.where(floating, melt, 0.0)


def integrate_by_basin(melt, area, floating, basin):
    """
    Integrate a melt rate over each basin, in Gt yr-1.

    Parameters
    ----------
    melt : xarray.DataArray
        Melt rate in kg m-2 yr-1

    area : xarray.DataArray
        Cell area in m2

    floating : xarray.DataArray
        True on floating cells

    basin : xarray.DataArray
        Basin number per cell

    Returns
    -------
    total : xarray.DataArray
        Integrated melt per basin, in Gt yr-1
    """
    weighted = (melt * area).where(floating)
    return weighted.groupby(basin.rename('basin')).sum() / 1.0e12


def interpolate_to_draft(field_3d, z_ocean, draft, bed):
    """
    Interpolate a 3-D ocean field to the ice draft.

    Written from the protocol rather than transliterated from MALI's
    Fortran, so that comparing the two is a real check of the
    implementation.  The four cases match MALI's: above the shallowest layer
    centre, below the deepest, where the layer below the draft is beneath
    the bed, and linear interpolation between layer centres.

    The freezing-point depth correction MALI applies below the deepest
    centre is *not* applied here, so this should only be used for fields
    where MALI does not apply it, or compared only over the interior cells.

    Parameters
    ----------
    field_3d : numpy.ndarray
        The field, shaped ``(nCells, nLayers)``

    z_ocean : numpy.ndarray
        Layer centre depths, negative downward and decreasing

    draft : numpy.ndarray
        Ice draft per cell, negative below sea level

    bed : numpy.ndarray
        Bed topography per cell

    Returns
    -------
    at_draft : numpy.ndarray
        The field interpolated to the draft, per cell
    """
    n_cells = field_3d.shape[0]
    at_draft = np.full(n_cells, np.nan)
    n_layers = len(z_ocean)

    for index in range(n_cells):
        # ksup is the deepest layer centre still at or above the draft
        above = np.nonzero(z_ocean >= draft[index])[0]
        ksup = above[-1] if above.size > 0 else -1
        if ksup < 0:
            at_draft[index] = field_3d[index, 0]
        elif ksup == n_layers - 1:
            at_draft[index] = field_3d[index, n_layers - 1]
        elif z_ocean[ksup + 1] < bed[index]:
            at_draft[index] = field_3d[index, ksup]
        else:
            span = z_ocean[ksup] - z_ocean[ksup + 1]
            w_deep = (z_ocean[ksup] - draft[index]) / span
            w_shallow = (draft[index] - z_ocean[ksup + 1]) / span
            at_draft[index] = (w_deep * field_3d[index, ksup + 1] +
                               w_shallow * field_3d[index, ksup])
    return at_draft
