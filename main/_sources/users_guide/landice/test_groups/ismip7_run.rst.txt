.. _landice_ismip7_run:

ismip7_run
==========

The ``landice/ismip7_run`` test group sets up one or more experiments from the
`ISMIP7 protocol <https://github.com/orgs/ismip/discussions>`_ for both the
Antarctic Ice Sheet (AIS) and the Greenland Ice Sheet (GrIS).

This functionality assumes the forcing files have already been generated using
the :ref:`landice_ismip7_forcing` test group and organized into the expected
directory layout. It creates a consistent set of run directories for the
requested experiments. Each experiment directory is self-contained with
namelists, streams, forcing symlinks, and a job script ready for submission.

The test group includes two test cases:

* ``ismip7_ais`` — Antarctic Ice Sheet experiments
* ``ismip7_gris`` — Greenland Ice Sheet experiments

.. note::

   This test group is not meant for automated running of experiments.
   Expert knowledge is recommended for conducting the actual simulations.
   Each experiment (step) should be submitted manually via its job script.

.. _landice_ismip7_run_experiments:

Experiment Matrix
-----------------

The ISMIP7 core protocol defines 11 experiments per ice sheet:

.. list-table::
   :header-rows: 1

   * - Experiment
     - Scenario
     - Start
     - End
     - ESM
   * - ``historical_CESM2-WACCM``
     - Historical
     - ≥1850
     - 2014
     - CESM2-WACCM
   * - ``historical_MRI-ESM2-0``
     - Historical
     - ≥1850
     - 2014
     - MRI-ESM2-0
   * - ``ssp370_CESM2-WACCM``
     - SSP370
     - 2015
     - 2100
     - CESM2-WACCM
   * - ``ssp370_MRI-ESM2-0``
     - SSP370
     - 2015
     - 2100
     - MRI-ESM2-0
   * - ``ssp126_CESM2-WACCM``
     - SSP126
     - 2015
     - 2300
     - CESM2-WACCM
   * - ``ssp126_MRI-ESM2-0``
     - SSP126
     - 2015
     - 2300
     - MRI-ESM2-0
   * - ``ssp585_CESM2-WACCM``
     - SSP585
     - 2015
     - 2300
     - CESM2-WACCM
   * - ``ssp585_MRI-ESM2-0``
     - SSP585
     - 2015
     - 2300
     - MRI-ESM2-0
   * - ``ctrl_CESM2-WACCM``
     - CTRL2015
     - 2015
     - 2300
     - CESM2-WACCM
   * - ``ctrl_MRI-ESM2-0``
     - CTRL2015
     - 2015
     - 2300
     - MRI-ESM2-0
   * - ``ocx``
     - OCX
     - 1990
     - 2025
     - (reanalysis)

Unlike ISMIP6, ISMIP7 requires a **separate historical simulation per ESM**.
Projection experiments automatically symlink their restart file from the
corresponding ESM's historical run (e.g.,
``ssp585_CESM2-WACCM`` → ``../historical_CESM2-WACCM/rst.2015-01-01.nc``).

.. _landice_ismip7_run_usage:

Usage
-----

1. Process forcing data using :ref:`landice_ismip7_forcing`.

2. Organize output into the expected directory layout::

      {forcing_basepath}/
      ├── CESM2-WACCM_historical/
      │   ├── atmosphere/
      │   │   └── {mesh}_smb_CESM2-WACCM_historical_*.nc
      │   ├── ocean_thermal_forcing/
      │   │   └── {mesh}_thermal_forcing_CESM2-WACCM_historical_*.nc
      │   └── shelf_collapse/
      │       └── {mesh}_ice_shelf_collapse_mask_*.nc
      ├── CESM2-WACCM_ssp585/
      │   ├── atmosphere/
      │   ├── ocean_thermal_forcing/
      │   └── shelf_collapse/
      └── ...

   The ``shelf_collapse`` subdirectory (ISMIP7 Path C ice shelf collapse
   mask, produced by :ref:`landice_ismip7_forcing_fracture`) is required
   for ``ssp*`` experiments unless ``use_hydrofracture_forcing`` is set to
   ``false``; see :ref:`landice_ismip7_run_mask_calving` below.

3. Create a user config file overriding the ``NotAvailable`` paths.

4. Set up and run::

      compass setup landice/ismip7_run/ismip7_ais -f my_ismip7_ais.cfg
      # Then submit job scripts from individual experiment directories

   If ``gia_model`` is set to ``1dSLM`` or ``FastIsostasy``, ``compass
   setup`` also adds a ``mapping_files`` or ``fastiso_mapping_files``
   subdirectory (respectively). These must be run (via their job script,
   or ``compass run`` from within that subdirectory) *before* submitting
   any individual experiment, since the experiments' job scripts assume
   the mapping files already exist.

.. _landice_ismip7_run_config:

config options
--------------

All config options should be reviewed and altered as needed.

**AIS config** (``[ismip7_run_ais]``):

.. code-block:: cfg

   [ismip7_run_ais]

   # Experiment list: "all", "historical", "projections", "ctrl",
   # or comma-delimited experiment names
   exp_list = all

   # Number of MPI tasks
   ntasks = 128
   pio_stride = 128

   # Base path to pre-processed forcing
   forcing_basepath = NotAvailable

    # Initial condition and parameter files
    init_cond_path = NotAvailable
    melt_params_path = NotAvailable
    reference_surface_path = NotAvailable
    region_mask_path = NotAvailable

   # Climatology files for CTRL2015 experiments
   ctrl_tf_climatology_path = NotAvailable
   ctrl_atm_climatology_path = NotAvailable

   # OCX forcing path
   ocx_forcing_path = NotAvailable

   # Calving: restore or von_mises
   calving_method = restore
   von_mises_parameter_path = NotAvailable

   # Whether to apply the ISMIP7 Path C ice shelf collapse mask as
   # hydrofracture-gated mask calving (not used for historical, ctrl, or
   # ocx experiments). If true, setup fails when the mask file is missing
   # for an experiment that should have one.
   use_hydrofracture_forcing = true

   # Ice fracture toughness (Pa m^0.5) for the hydrofracture vulnerability
   # criterion gating mask calving, used only when use_hydrofracture_forcing
   # is true.
   calving_fracture_toughness = 2.0e5

   # Face melting
   use_face_melting = false

   # Glacial isostatic adjustment (GIA) coupling: none, 1dSLM, or FastIsostasy
   gia_model = none

   # Sea-level model coupling (used if gia_model = 1dSLM)
   slm_input_ice = NotAvailable
   slm_input_earth = NotAvailable
   slm_earth_structure = prem_512.l60K2C.sum18p6.dum19p2.tz19p4.lm22
   slm_input_others = NotAvailable
   nglv = 2048

   # FastIsostasy (regional bedrock/GIA) coupling (used if
   # gia_model = FastIsostasy)
   fastiso_path = NotAvailable
   fastiso_earth_structure_filename = weak-earth.nc
   fastiso_mask_filename = None
   fastiso_res_km = 8

.. note::

   ``fastiso_res_km`` sets the resolution (in km) of the regular grid
   FastIsostasy runs on, which should match one of the standard ISMIP7
   grid resolutions (2, 4, 8, or 16 km for AIS). As of this writing,
   resolutions finer than 8 km have been found to yield numerically
   unstable results.

.. note::

   ``fastiso_path`` is a directory holding both the Earth-structure
   (rheology) file named by ``fastiso_earth_structure_filename`` and,
   optionally, an interactive-sea-level activation mask file named by
   ``fastiso_mask_filename``. Leaving ``fastiso_mask_filename = None``
   uses FastIsostasy's built-in fallback, which activates interactive
   sea level only over the un-padded ice domain -- a hard cutoff at the
   domain/padding boundary that can show up as a ring-shaped artifact in
   bed-topography change near the edge of the ice sheet. Supplying a mask
   file (NetCDF with dims ``x``/``y``, 1-D coordinate variables ``x``/``y``
   in meters, and a 2-D variable ``M``) lets you activate it over a wider
   region, e.g. the full padded domain, instead.

**GrIS config** (``[ismip7_run_gris]``) is similar (including
``reference_surface_path``) but without sea-level model or FastIsostasy
coupling options

.. _landice_ismip7_run_forcing_streams:

Forcing Streams
---------------

ISMIP7 uses more forcing fields than ISMIP6, at mixed temporal resolutions:

**Monthly forcing** (``input_interval = 0000-01-00_00:00:00``):

* ``sfcMassBal`` — surface mass balance
* ``surfaceAirTemperature`` — surface air temperature
* ``ismip6Runoff`` — ice sheet runoff
* ``ismip6_2dThermalForcing`` (GrIS) — ocean thermal forcing

**Annual forcing** (``input_interval = 0001-00-00_00:00:00``):

* ``sfcMassBalLapseRate`` — SMB elevation lapse rate
* ``surfaceAirTemperatureLapseRate`` — temperature lapse rate
* ``ismip6shelfMelt_3dThermalForcing`` (AIS)

**Static** (``input_interval = initial_only``):

* ``ismip6shelfMelt_zOcean`` — ocean depth coordinates (AIS only)
* ``ismip6shelfMelt_deltaT``, ``ismip6shelfMelt_basin``,
  ``ismip6shelfMelt_gamma0`` — melt parameterization coefficients

For CTRL2015 experiments, all forcing intervals are set to
``initial_only`` (constant climate).

.. _landice_ismip7_run_mask_calving:

Mask Calving (Path C)
----------------------

``ismip7_ais`` can optionally apply the ISMIP7 Path C ice shelf collapse
mask produced by :ref:`landice_ismip7_forcing_fracture`. The mask is looked
up under the same ``forcing_basepath`` used for the atmosphere/ocean
forcing, at
``{forcing_basepath}/{model}_{scenario}/shelf_collapse/*ice_shelf_collapse_mask_*.nc``.
``historical``, ``ctrl``, and ``ocx`` experiments never use this pathway,
regardless of ``use_hydrofracture_forcing``.

When ``use_hydrofracture_forcing`` is ``true`` (the default) for any other
experiment (e.g. ``ssp126``, ``ssp370``, ``ssp585``), a matching mask file
is required: if it is missing, ``compass setup`` fails with an error rather
than silently skipping mask calving. When found, the mask is symlinked into
the run directory and used to force calving:

* ``config_apply_calving_mask`` is set to ``.true.``
* ``config_restore_calving_front`` is set to ``.false.``
* ``config_require_extensional_stresses_for_mask_calving`` is set to
  ``.true.``, so the mask only removes floating ice that is also vulnerable
  to hydrofracture per the fracture-toughness criterion of Lai et al.
  (2020) / Reynolds and Nowicki (2026). The threshold depends on
  ``config_calving_fracture_toughness``.

Set ``use_hydrofracture_forcing = false`` to disable mask calving and the
hydrofracture gating entirely; experiments then use their normal
``calving_method`` configuration instead.

.. _landice_ismip7_run_ais:

ismip7_ais
----------

``landice/ismip7_run/ismip7_ais`` sets up AIS experiments with 3D ocean
thermal forcing (30 vertical layers) and optional sea-level model and/or
FastIsostasy coupling.

.. _landice_ismip7_run_gris:

ismip7_gris
-----------

``landice/ismip7_run/ismip7_gris`` sets up GrIS experiments with 2D
(depth-averaged) ocean thermal forcing. Sea-level model coupling is not
currently supported for GrIS. Crevasse-depth calving is the default.
