.. _dev_landice_ismip7_forcing:

ismip7_forcing
==============

The ``ismip7_forcing`` test group
(:py:class:`compass.landice.tests.ismip7_forcing.Ismip7Forcing`) processes
(i.e., remaps and renames) the atmospheric and ocean thermal forcing data of
the Ice Sheet Model Intercomparison for CMIP7 (ISMIP7) protocol from its
native polar stereographic grid to the MALI unstructured mesh. The test group
supports both AIS and GrIS via the ``ice_sheet`` config option. It includes
three test cases: ``atmosphere``, ``ocean_thermal``, and ``fracture``.

.. _dev_landice_ismip7_forcing_framework:

framework
---------

The shared config options for the ``ismip7_forcing`` test group are described
in :ref:`landice_ismip7_forcing` in the User's Guide.

ice_sheet_params
~~~~~~~~~~~~~~~~

The module :py:mod:`compass.landice.tests.ismip7_forcing.ice_sheet_params`
defines a dictionary of ice-sheet-specific parameters (projection, file
naming prefix, grid resolution, data version, ocean dimensionality, and the
atmosphere/ocean source names) and provides the function
:py:func:`compass.landice.tests.ismip7_forcing.ice_sheet_params.get_params`
to retrieve them based on the ``ice_sheet`` config option.

When ``scenario = OCX``, ``get_params`` applies a set of OCX overrides on top
of the ice-sheet defaults: data version ``v1``, the ocean file-name grid token
(e.g. ``ocean-1000m``), and the fixed reanalysis sources (``atm_model`` =
``RACMO2.3p2-ERA`` and ``ocean_model`` = ``EN4``). The processing steps use
``atm_model`` / ``ocean_model`` in place of the ``[ismip7] model`` option when
they are set, so the OCX ``model`` option is ignored. This keeps OCX handling
centralized and lets a single config file drive both test cases.

configure
~~~~~~~~~

The module :py:mod:`compass.landice.tests.ismip7_forcing.configure` validates
that all required config options in the ``[ismip7]`` section have been set by
the user (i.e., are not ``NotAvailable``).

Repository-local example user configs are available at
``compass/landice/tests/ismip7_forcing/ismip7_forcing_test.cfg`` (AIS) and
``compass/landice/tests/ismip7_forcing/ismip7_forcing_test_gis.cfg`` (GrIS).
The GrIS OCX scenario has a dedicated example
``ismip7_forcing_ocx_gis.cfg`` in the same directory.
These are intended for development/testing and include environment-specific
paths.

create_mapfile
~~~~~~~~~~~~~~

The module :py:mod:`compass.landice.tests.ismip7_forcing.create_mapfile`
defines a unified framework for creating SCRIP and mapping files. The function
:py:func:`compass.landice.tests.ismip7_forcing.create_mapfile.build_mapping_file`
creates a SCRIP file from the input polar stereographic grid using the
``create_scrip_file_from_planar_rectangular_grid`` command from MPAS-Tools,
then generates a mapping file via ``ESMF_RegridWeightGen``. The projection
is automatically determined from the ``ice_sheet`` config option using
``ice_sheet_params``.

Test cases
----------

.. _dev_landice_ismip7_forcing_atmosphere:

atmosphere
~~~~~~~~~~

The :py:class:`compass.landice.tests.ismip7_forcing.atmosphere.Atmosphere`
test case processes the ISMIP7 atmosphere forcing fields. It contains five
steps: SMB, temperature, their respective gradients, and runoff. Each step
discovers input files matching the ice-sheet-specific naming pattern, builds
or reuses a mapping file, remaps each input file with ``ncremap``, and
combines/renames the results to MALI conventions.

Steps:

* :py:class:`~compass.landice.tests.ismip7_forcing.atmosphere.process_smb.ProcessSmb` —
  ``acabf`` → ``sfcMassBal``
* :py:class:`~compass.landice.tests.ismip7_forcing.atmosphere.process_temperature.ProcessTemperature` —
  ``ts`` → ``surfaceAirTemperature`` (clipped ≤ 273.15 K)
* :py:class:`~compass.landice.tests.ismip7_forcing.atmosphere.process_smb_gradient.ProcessSmbGradient` —
  ``dacabfdz`` → ``sfcMassBalLapseRate``
* :py:class:`~compass.landice.tests.ismip7_forcing.atmosphere.process_temperature_gradient.ProcessTemperatureGradient` —
  ``dtsdz`` → ``surfaceAirTemperatureLapseRate``
* :py:class:`~compass.landice.tests.ismip7_forcing.atmosphere.process_runoff.ProcessRunoff` —
  ``mrro`` → ``ismip6Runoff``

.. _dev_landice_ismip7_forcing_ocean_thermal:

ocean_thermal
~~~~~~~~~~~~~

The :py:class:`compass.landice.tests.ismip7_forcing.ocean_thermal.OceanThermal`
test case processes the ISMIP7 ocean thermal forcing. It contains two steps,
:py:class:`~compass.landice.tests.ismip7_forcing.ocean_thermal.process_thermal_forcing.ProcessThermalForcing`,
which handles both AIS (3D, decade-spanning files) and GrIS (2D, yearly files)
by branching on the ``ocean_3d`` parameter from ``ice_sheet_params``, and
:py:class:`~compass.landice.tests.ismip7_forcing.ocean_thermal.build_3d_thermal_forcing.BuildGreenland3dThermalForcing`,
which optionally builds a 3D GrIS field from the 2D forcing (see below).

The ``run()`` method dispatches to two sub-methods based on the boolean config
options ``process_ocean_thermal`` and ``process_ocean_climatology`` in the
``[ismip7]`` section:

* ``_run_scenario()``: Processes time-varying ESM scenario data (model +
  scenario combination). Uses config from ``[ismip7_ocean_thermal]``.
* ``_run_climatology()``: Processes the static observational climatology
  (Zhou et al., AIS only). Uses config from ``[ismip7_ocean_climatology]``.
  The TF version (currently v3) is hard-coded.

For AIS scenario data, the step:

* Remaps thermal forcing preserving 30 vertical ocean layers
* Produces ``ismip6shelfMelt_3dThermalForcing`` (dims: Time × nCells ×
  nISMIP6OceanLayers)
* Includes depth coordinate variables ``ismip6shelfMelt_zOcean`` and
  ``ismip6shelfMelt_zBndsOcean``

For AIS climatology data, the step:

* Extrapolates fill values, remaps, and renames to MALI conventions
* Produces ``ismip6shelfMelt_3dThermalForcing`` (dims: nCells ×
  nISMIP6OceanLayers) — no Time dimension

For GrIS, the step:

* Remaps 2D monthly thermal forcing
* Produces ``ismip6_2dThermalForcing`` (dims: Time × nCells)

build_3d_thermal_forcing (GrIS 3-D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The
:py:class:`~compass.landice.tests.ismip7_forcing.ocean_thermal.build_3d_thermal_forcing.BuildGreenland3dThermalForcing`
step (GrIS only, gated by ``process_ocean_thermal_3d``) converts the GrIS 2D
thermal forcing into a 30-level 3D field for MALI's nonlocal (Jourdain et al.
2020) melt scheme. Its ``run()`` reconstructs the 2D output path that
``ProcessThermalForcing._run_scenario`` wrote (mirroring the ``forcing_group``
and ocean-source logic), injects the compass-derived mesh / 2D-forcing /
output / diagnostics paths into a
:py:class:`~compass.landice.tests.ismip7_forcing.ocean_thermal.greenland_3d.Config`
built from the JSON ``config_file``, and calls
:py:func:`~compass.landice.tests.ismip7_forcing.ocean_thermal.greenland_3d.run`.
The ported science lives in
:py:mod:`compass.landice.tests.ismip7_forcing.ocean_thermal.greenland_3d`:
seven regional EN4 profiles, seafloor anchoring to the 2D forcing, and
per-region ``deltaT`` calibration. The output supplements the 2D file with
``ismip6shelfMelt_3dThermalForcing``, ``ismip6shelfMelt_deltaT``,
``ismip6shelfMelt_gamma0``, ``ismip6shelfMelt_zOcean``, and
``ismip6shelfMelt_basin``. The multi-gigabyte field is streamed record by
record (dask, ``scipy`` engine, ``NETCDF3_64BIT``, ``.partial``-then-rename).

.. _dev_landice_ismip7_forcing_fracture:

fracture
~~~~~~~~

The :py:class:`compass.landice.tests.ismip7_forcing.fracture.Fracture`
test case processes the ISMIP7 surface-melt-driven ice shelf collapse
forcing (AIS only). It implements the three ISMIP7 pathways as independent
steps, each discovering its source file from the ``fracture/{version}/``
subdirectory of ``base_path_ismip7``, building or reusing a mapping file,
remapping with ``ncremap``, and renaming the result to MALI conventions with
an accompanying ``xtime`` variable. Per-pathway remapping methods are set in
the ``[ismip7_fracture]`` config section. Setting a pathway's remapping-method
option to ``None`` causes that step to return early without processing its
file, which is useful when only some pathway source files are available.

Steps:

* :py:class:`~compass.landice.tests.ismip7_forcing.fracture.process_excess_melt.ProcessExcessMelt`
  (Path A) — ``excess_melt`` → ``ismip7ExcessMelt``. The excess melt file
  lacks ``x``/``y`` coordinate variables and its array is flipped along the
  y axis relative to the other fracture files (it was produced with CDO).
  The ``_prepare_source_grid()`` method borrows ``x``/``y`` from a sibling
  fracture file, flips the data to match (raising if the flipped ``lat`` does
  not match the sibling grid), and writes a reconstructed source file. The
  field is then extrapolated (nearest neighbor, filling NaNs) and remapped
  conservatively by default (it is a flux).
* :py:class:`~compass.landice.tests.ismip7_forcing.fracture.process_lake_properties.ProcessLakeProperties`
  (Path B) — ``lake_depth`` → ``ismip7LakeDepth`` and
  ``fraction_lake_area`` → ``ismip7LakeAreaFraction``. Both variables are
  extrapolated and remapped in a single ``ncremap`` call (bilinear by
  default).
* :py:class:`~compass.landice.tests.ismip7_forcing.fracture.process_shelf_collapse.ProcessShelfCollapse`
  (Path C) — ``mask`` → ``calvingMask``. Remapped with ``neareststod`` by
  default and rounded to 0/1 so the discrete collapse mask is preserved.

The annual source fields use an integer ``year``/``time`` coordinate with
``units="year"`` (not CF-compliant), so each step opens the data with
``decode_times=False`` and constructs ``xtime`` at January 1st of each year.

Shared remapping helpers used by the fracture steps live in
:py:mod:`compass.landice.tests.ismip7_forcing.fracture.remap_utils`:
``extrapolate_source`` (nearest-neighbor fill of NaNs on the source grid),
``open_rename_and_trim`` (open a remapped file, rename dimensions/variables
to MALI conventions, and restrict to the requested year range), and
``add_xtime_and_write`` (add the ``xtime`` variable, drop auxiliary remapping
variables, and write the output).

The output variable names for Paths A and B (``ismip7ExcessMelt``,
``ismip7LakeDepth``, ``ismip7LakeAreaFraction``) are descriptive placeholders
and may need to be aligned with the MALI Registry once the corresponding
model input fields are defined.
