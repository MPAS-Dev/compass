.. _dev_landice_ismip7_calibration:

ismip7_calibration
==================

The ``ismip7_calibration`` test group
(:py:class:`compass.landice.tests.ismip7_calibration.Ismip7Calibration`)
calibrates MALI's sub-shelf melt parameterization against the ISMIP7
Antarctic ice-ocean protocol (Reese et al., Sect. 4.2).

The protocol asks each ice-sheet model to fit the free parameter of its melt
module against four objective-function terms and to report the 5th, 50th and
95th percentiles of the resulting parameter distribution:

===== ==========================================================
term  what it constrains
===== ==========================================================
J1    present-day melt integrated over each IMBIE2 drainage basin
J2    present-day melt integrated over bins of equal buttressing importance
J3    the warm-minus-cold basin-mean melt difference of ocean models
J4    observed melt of Pine Island and Dotson, per observation year
===== ==========================================================

The user's guide describes the config options in
:ref:`landice_ismip7_calibration`.

The test group has two test cases: ``replication`` and ``ais``.

.. _dev_landice_ismip7_calibration_framework:

framework
---------

Code shared with the other ISMIP7 test groups lives in the landice framework
package :py:mod:`compass.landice.ismip7`, described in
:ref:`dev_landice_framework`.  The modules below are specific to the
calibration.

datasets
~~~~~~~~

:py:mod:`compass.landice.tests.ismip7_calibration.datasets` is the single
registry of *which* ocean states feed *which* objective-function term, and of
where the calibration targets live.  Both test cases drive it, so the 8 km
replication and the MALI-mesh calibration cannot drift apart.

:py:func:`compass.landice.tests.ismip7_calibration.datasets.ocean_states`
returns the ``minimal`` (7), ``recommended`` (11) or ``all`` (28) subset of
protocol Table 2.  Regional ocean models record the basins they cover, so
that melt outside their domains is discarded.

terms
~~~~~

:py:mod:`compass.landice.tests.ismip7_calibration.terms` implements J1-J4 in
a form that works on any mesh.  The upstream toolbox assumes a uniform
structured grid: it converts cell sums to Gt/yr with a single scalar
``reso**2`` and takes unweighted means over basins.  That is correct on the
ISMIP 8 km grid and wrong on MALI's 4-20 km mesh, where cell area varies by a
factor of about 25.  These functions take an explicit cell-area array
instead, so passing a uniform area reproduces the structured-grid answer
exactly while a real mesh gets an area-weighted one.

quadratic
~~~~~~~~~

:py:mod:`compass.landice.tests.ismip7_calibration.quadratic` implements both
melt formulas in Python: the Burgard et al. (2022) local quadratic of
protocol Eq. (1), and the ISMIP6 non-local quadratic.  It serves as the
specification MALI's Fortran is checked against and as what the replication
drives.

Note that ``local_quadratic_melt`` takes the slope *angle* while MALI's
``config_ismip7_melt_sin_slope`` is a *sine*;
:py:func:`compass.landice.tests.ismip7_calibration.quadratic.angle_from_sin_slope`
converts between them.

toolbox
~~~~~~~

:py:mod:`compass.landice.tests.ismip7_calibration.toolbox` holds a verbatim
copy of the upstream ISMIP7 parameter-selection toolbox, which implements the
objective function itself.  ``PROVENANCE.md`` alongside it records the
upstream commit and a SHA256 that is verified on import and asserted by a
unit test, so an accidental edit or a partial update fails loudly rather than
quietly changing published numbers.

objective
~~~~~~~~~

:py:mod:`compass.landice.tests.ismip7_calibration.objective` assembles the
toolbox's arguments and reduces its output to percentiles.  Each term is
built once at a unit parameter value and scaled onto the parameter grid,
which is exact because melt is proportional to the parameter.

.. _dev_landice_ismip7_calibration_replication:

replication
-----------

``landice/ismip7_calibration/replication`` reproduces the published 8 km
calibration through compass's own code path, using the Python reference melt
implementation on the ISMIP grid.  It runs in about a minute, needs no MALI
run, and must reproduce the published percentiles

.. code-block:: none

    K = 4.75e-5 / 8.5e-5 / 1.375e-4

exactly.  ``validate()`` raises if it does not.  This is the check that the
vendored toolbox is being driven correctly; the MALI calibration is built on
the same code path, so if this fails the MALI numbers cannot be trusted
either.

This test case deliberately uses the **published inputs**, including the
spatially varying 8 km salinity fields.  It replicates a published
calculation, so it must not be changed to match the constant-salinity choice
the MALI calibration makes.

.. _dev_landice_ismip7_calibration_ais:

ais
---

``landice/ismip7_calibration/ais`` is the calibration itself, on an Antarctic
MALI mesh.

``remap_masks``
    Remaps the ISMIP7 IMBIE2 basins, buttressing (BFRN) bins, floating mask
    and PIG/Dotson regions onto the MALI mesh, nearest-neighbour throughout
    since every field is categorical.

    **Two basin-numbering conventions are in play and they differ by one.**
    ISMIP7's ``basinNumber`` is 0-based, 0-15, with basin 9 the Eastern
    Amundsen and basin 14 Ronne-Filchner, which is how the protocol refers to
    them.  MALI's ``ismip6shelfMelt_basin`` is 1-based, 1-16, so MALI basin
    10 is ISMIP7 basin 9.  Both are written, under distinct names, so that
    neither can be silently reinterpreted as the other -- a mistake that
    mis-assigns every basin while still producing plausible-looking numbers.
    The step cross-tabulates the result against the mesh's existing
    ``regionCellMasks`` in both directions and fails below 80% agreement.

``remap_forcing``
    Remaps the calibration thermal forcing for each ocean state.  The
    observational states cover only about 6% of the ISMIP grid (protocol
    Sect. A10), so they are filled from the present-day climatology *before*
    remapping, exactly as the regional model datasets are distributed
    (Sect. A9); interpolation then never blends a real value with a missing
    one.  The remapped field is required to be finite everywhere.

    **Only thermal forcing is remapped.**  See the salinity note below.

``<melt_form>_<ocean_state>``
    One single-timestep MALI melt diagnostic per ocean state and melt form,
    with the velocity solver off, so Albany is not required.

    The run takes **one short timestep rather than none**: MALI computes melt
    inside the timestep, not in the initial diagnostic solve, so a zero-length
    run would produce no melt at all.

    There is one run per ocean state, **not** one per parameter value.  Melt
    is exactly proportional to the melt parameter, so the whole parameter
    ensemble follows by scaling a single run.  That is what makes this 28 runs
    per melt form rather than about 1300.  Do not "improve" this away.

``verify_melt``
    Checks MALI's melt against the Python reference evaluated on MALI's *own*
    ``TFdraft``, which isolates the melt expression from the vertical
    interpolation; checks that interpolation against an independent
    implementation written from the protocol; and measures the linearity the
    ensemble design relies on.

    The draft is reconstructed from the mesh file, never taken from the run
    output.  Melt thins the ice over the single timestep, so the output
    ``lowerSurface`` and ``thickness`` are *post*-step while ``TFdraft`` was
    computed *pre*-step; pairing them is inconsistent by up to a metre of
    draft.

``aggregate``
    Aggregates melt to basins, buttressing bins and shelf regions.  All
    aggregation is **area-weighted**: integrals use ``melt * areaCell`` and
    means are weighted by ``areaCell``.  Also reports MALI's per-basin
    ice-shelf area against the observed ISMIP7 extent, since J1, J2 and J4
    are integrals and any shelf-area mismatch enters the calibrated parameter
    directly.

``calibrate``
    Runs the 100,000-sample parameter selection per melt form.  The objective
    **normalises each term by its own median** over the parameter ensemble, so
    a minimised objective is *not* comparable between melt forms -- only
    between parameter values within one form.

``fit_delta_t``
    Fits the per-basin thermal-forcing correction ``dT_b`` **after** parameter
    selection, following protocol Sect. 4.2.1 option 2, as the published
    quadratic worked example does.  Fitting it first would leave the parameter
    bounds unconstrained where present-day melt is compared with observations,
    which is the drawback Sect. 4.2.1 names.  The toolbox's own
    ``optimise_deltaT`` cannot be used because it sums over structured-grid
    ``x`` and ``y`` with a single scalar cell area, so the search is
    reimplemented area-weighted.  ``dT`` enters the melt only through the
    thermal forcing, so melt is recomputed in Python from ``TFdraft`` and no
    further MALI runs are needed.

``report``
    Parameter-distribution plots and a summary table.

melt forms and salinity
-----------------------

Two melt forms are calibrated, selected with the ``melt_forms`` config
option:

``ismip7``
    The Burgard et al. (2022) **local** quadratic of protocol Eq. (1),
    calibrating ``K``.

``ismip6``
    The **non-local** quadratic MALI already had, calibrating ``gamma0``.
    With a constant salinity this is algebraically identical to the Burgard
    *semi-local* form of protocol Eq. (2) -- only the decomposition of the
    constant differs -- so this is how the semi-local form is calibrated,
    rather than as a third melt module.

**The calibration uses a constant salinity**, set by the ``salinity`` config
option and passed to MALI as
``config_ismip7_melt_salinity_source = 'constant'``.  MALI can read a 3-D
salinity field, but the ISMIP7 ocean forcing already processed for the MALI
projections carries thermal forcing only, so a projection has no salinity
field to read.  Calibrating against a spatially varying salinity would tune
``K`` for physics the projections cannot run.

Melt is linear in salinity and basin-mean salinity spans about 34.1 to 34.7
against the constant 34.5, so this shifts the calibrated parameter by roughly
1%.  Note that the published ``K`` was obtained with a locally varying
salinity, so comparison with it is approximate at about that level.

using a MALI build with the ISMIP7 melt method
----------------------------------------------

The ``ismip7`` melt form needs a MALI build that has
``config_basal_mass_bal_float = 'ismip7'``.  If the compass ``MALI-Dev``
submodule does not yet have it, point compass at another build with

.. code-block:: cfg

    [paths]
    mpas_model = /path/to/E3SM/components/mpas-albany-landice

Compass only *warns* when a namelist option is missing from the model's
defaults, so a build without the ISMIP7 melt method would silently drop
``config_ismip7_melt_K`` and produce a plausible but wrong calibration.  The
``run_state`` step therefore checks the default namelist at setup and fails
with a clear message instead.
