"""
The ISMIP7 "quadratic local" melt parameterisation, Burgard et al. (2022).

Protocol Eq. (1):

.. code-block:: none

    m = K sin(theta) (rho_o/rho_i) (c_o/L_i)^2 beta_S S_loc
        (g/(2|f|)) |TF_loc| TF_loc

This is the reference implementation used both to replicate the published
8 km calibration (see the ``replication`` test case) and as the specification
for MALI's Fortran implementation, selected with
``config_basal_mass_bal_float = 'ismip7'``.

It is written to agree bit-for-bit with ``multimelt.melt_functions
.quadratic_mixed_slope``, which is what the protocol's own worked example uses,
so that any difference in the calibrated ``K`` is attributable to the mesh and
the ice-sheet model rather than to the melt formula.

Note on the Coriolis parameter: multimelt uses a *constant* ``f = 1.4e-4``
rather than a latitude-varying one.  The MALI mesh carries ``latCell``, so a
latitude-varying ``f`` would be easy, but it would shift ``K`` away from the
published value.  The constant is deliberate, so that ``K`` stays comparable.
"""

from dataclasses import dataclass

import numpy as np

#: seconds per year, as used by multimelt
SECONDS_PER_YEAR = 31556926.080000002

#: (rho_sw * c_pw) / (rho_i * L_i), K^-1 -- multimelt's ``melt_factor``
MELT_FACTOR = 0.013338444158574889

#: specific heat capacity of seawater, J kg^-1 K^-1
C_PO = 3974.0

#: latent heat of fusion of ice, J kg^-1
L_I = 334000.0

#: haline contraction coefficient, from Lazeroms et al.
BETA_S = 0.000786

#: gravitational acceleration, m s^-2
GRAVITY = 9.81

#: Coriolis parameter, s^-1 (constant, as in multimelt)
F_CORIOLIS = 0.00014

#: ice density used by the protocol's worked example, kg m^-3
ICE_DENSITY = 918.0


@dataclass(frozen=True)
class Constants:
    """
    The physical constants entering protocol Eq. (1).

    Two sets matter here.  ``MULTIMELT`` reproduces the reference
    implementation the published calibration used, and ``MALI`` uses the values
    MALI itself compiles in, so that a MALI melt field can be checked against
    this formula without the constants confounding the comparison.

    Attributes
    ----------
    c_o : float
        Specific heat capacity of seawater, J kg-1 K-1.
    latent_heat : float
        Latent heat of fusion of ice, J kg-1.
    gravity : float
        Gravitational acceleration, m s-2.
    beta_s : float
        Haline contraction coefficient, PSU-1.
    coriolis : float
        Coriolis parameter magnitude, s-1.
    rho_ice : float
        Ice density appearing in the (rho_o / rho_i) factor of Eq. (1),
        kg m-3.
    rho_ice_flux : float
        Ice density used to convert a melt rate in m of ice to a mass flux,
        kg m-3.  Normally identical to ``rho_ice``, in which case the two
        cancel and the result is independent of ice density.  They differ in
        the protocol's worked example, which takes 917 in the formula and 918
        in the conversion, leaving a spurious factor of 918/917.
    rho_ocean : float
        Seawater density, kg m-3.
    seconds_per_year : float
        Year length used to report the melt rate per year.  The melt rate
        itself is per second and unambiguous; this only sets the units it is
        reported in.  MALI uses a 365-day year, matching its noleap calendar,
        while the protocol's reference implementation uses 365.2422 days -- a
        0.066% difference, which is enough to fail an exact comparison.
    """

    c_o: float
    latent_heat: float
    gravity: float
    beta_s: float
    coriolis: float
    rho_ice: float
    rho_ice_flux: float
    rho_ocean: float
    seconds_per_year: float


#: constants of the protocol's reference implementation (multimelt)
MULTIMELT = Constants(
    c_o=3974.0,
    latent_heat=334000.0,
    gravity=9.81,
    beta_s=0.000786,
    coriolis=0.00014,
    rho_ice=917.0,
    rho_ice_flux=918.0,
    rho_ocean=1028.0,
    seconds_per_year=SECONDS_PER_YEAR,
)

#: constants MALI compiles in, from li_constants and the default namelist
MALI = Constants(
    c_o=3.974e3,
    latent_heat=335.0e3,
    gravity=9.80616,
    beta_s=7.86e-4,
    coriolis=1.4e-4,
    rho_ice=910.0,
    rho_ice_flux=910.0,
    rho_ocean=1028.0,
    # li_constants scyr: seconds in a 365-day year
    seconds_per_year=31536000.0,
)


def u_factor(salinity):
    """
    The velocity-scale factor of Jenkins et al. (2018).

    ``(c_o / L_i) * beta_S * g / (2 |f|) * S_loc``

    Parameters
    ----------
    salinity : xarray.DataArray or float
        Practical salinity at the ice draft.

    Returns
    -------
    Same type as ``salinity``.
    """
    return (
        (C_PO / L_I) * BETA_S * (GRAVITY / (2.0 * abs(F_CORIOLIS))) * salinity
    )


def local_quadratic_melt(
    k,
    thermal_forcing,
    salinity,
    slope,
    thermal_forcing_avg=None,
    constants=None,
    delta_t=0.0,
):
    """
    Melt rate from the quadratic local parameterisation, in kg m-2 yr-1.

    Parameters
    ----------
    k : float or xarray.DataArray
        The calibration parameter ``K``.
    thermal_forcing : xarray.DataArray
        Local thermal forcing at the ice draft, in K.
    salinity : xarray.DataArray
        Practical salinity at the ice draft.
    slope : float or xarray.DataArray
        Ice-draft slope angle in radians (positive).  A scalar gives the
        "constant Antarctic-mean slope" variant; a field gives the
        slope-dependent one.
    delta_t : float or xarray.DataArray, optional
        Basin-wide thermal-forcing correction, K.  Protocol §4.2.1 applies it
        wherever the thermal forcing appears, so it is added to both the local
        forcing and the averaged one.  Defaults to zero, which is what the
        calibration uses; production runs are expected to use non-zero values.
    thermal_forcing_avg : xarray.DataArray, optional
        Thermal forcing to use in the ``|TF|`` factor.  Defaults to
        ``thermal_forcing``, giving the *local* form; pass a shelf- or
        basin-average for the *semi-local* form of protocol Eq. (2).

    Returns
    -------
    xarray.DataArray
        Melt rate in kg m-2 yr-1, positive for melting.

    Notes
    -----
    Melt is exactly linear in ``k``, which is what allows the calibration to
    use one model run per ocean state rather than one per (state, parameter)
    pair; see the linearity of melt in the parameter.
    """
    if thermal_forcing_avg is None:
        thermal_forcing_avg = thermal_forcing
    thermal_forcing = thermal_forcing + delta_t
    thermal_forcing_avg = thermal_forcing_avg + delta_t

    if constants is None:
        melt_factor = MELT_FACTOR
        u = u_factor(salinity)
        rho_ice = ICE_DENSITY
        seconds_per_year = SECONDS_PER_YEAR
    else:
        melt_factor = (
            constants.rho_ocean * constants.c_o /
            (constants.rho_ice * constants.latent_heat))
        u = (
            (constants.c_o / constants.latent_heat) *
            constants.beta_s *
            (constants.gravity / (2.0 * abs(constants.coriolis))) *
            salinity)
        rho_ice = constants.rho_ice_flux
        seconds_per_year = constants.seconds_per_year

    melt_m_per_s = (
        k * melt_factor * u * thermal_forcing *
        abs(thermal_forcing_avg) * np.sin(slope))
    return melt_m_per_s * seconds_per_year * rho_ice


def draft_slope(draft, dx, dy, x_dim='x', y_dim='y'):
    """
    Ice-draft slope angle on a structured grid, in radians.

    Reproduces the centred-difference scheme of
    ``multimelt.plume_functions.check_slope_one_dimension``, including its
    one-sided fallbacks at NaN neighbours and its substitution of zero where
    the slope cannot be computed at all.

    Parameters
    ----------
    draft : xarray.DataArray
        Ice draft (negative below sea level), on a regular grid.
    dx, dy : float
        Grid spacing in m.
    x_dim, y_dim : str, optional
        Names of the horizontal dimensions.

    Returns
    -------
    xarray.DataArray
        Slope angle in radians.
    """
    slope_x = _one_dimensional_slope(draft, x_dim, dx)
    slope_y = _one_dimensional_slope(draft, y_dim, dy)
    return np.arctan(np.sqrt(slope_x**2 + slope_y**2))


def _one_dimensional_slope(draft, dim, spacing):
    """One-sided-tolerant centred difference, as in multimelt."""
    shifted_minus = draft.shift({dim: -1})
    shifted_plus = draft.shift({dim: 1})

    both = (shifted_minus - shifted_plus) / np.sqrt((2.0 * spacing) ** 2)
    right = (draft - shifted_plus) / np.sqrt(spacing**2)
    left = (shifted_minus - draft) / np.sqrt(spacing**2)

    slope = both.combine_first(right).combine_first(left)
    return slope.where(np.isfinite(slope), 0.0)


def mean_slope(draft, floating, dx, dy, x_dim='x', y_dim='y'):
    """
    Antarctic-mean ice-draft slope angle over floating ice, in radians.

    This is the "constant slope" used by the protocol's worked example.  Note
    that it is a mean of *angles*, not of ``sin(theta)``, and that its value is
    resolution dependent; both points are raised in
    ``protocol-and-toolbox-questions.md`` A4.

    Parameters
    ----------
    draft : xarray.DataArray
        Ice draft on a regular grid.
    floating : xarray.DataArray
        Boolean mask, True on floating ice.
    dx, dy : float
        Grid spacing in m.
    x_dim, y_dim : str, optional
        Names of the horizontal dimensions.

    Returns
    -------
    float
        Mean slope angle in radians.
    """
    slope = draft_slope(draft, dx, dy, x_dim=x_dim, y_dim=y_dim)
    return float(slope.where(floating).mean())
