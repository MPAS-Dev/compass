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

Code shared with the other ISMIP7 test groups lives in the landice framework
package :py:mod:`compass.landice.ismip7`, described in
:ref:`dev_landice_framework`.  This test group uses
:py:func:`compass.landice.ismip7.ice_sheet_params.get_params` for invariant
ice-sheet-specific parameters (projection, file naming prefix, and ocean
dimensionality), :py:mod:`compass.landice.ismip7.archive` for native archive
path, resolution, and version discovery,
:py:func:`compass.landice.ismip7.mapping.build_mapping_file` to create the
SCRIP and ESMF mapping files, and the remapping helpers in
:py:mod:`compass.landice.ismip7.remap`.

When ``scenario = OCX``, ``get_params`` supplies the fixed reanalysis sources
(``atm_model`` = ``RACMO2.3p2-ERA`` and ``ocean_model`` = ``EN4``). The
archive resolver discovers their native paths, selected resolutions, and
versions. The processing steps use
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
combines/renames the results to MALI conventions. ``latest`` is resolved per
variable because atmosphere datasets can have different current versions.

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
test case processes the ISMIP7 ocean thermal forcing. It contains a single step,
:py:class:`~compass.landice.tests.ismip7_forcing.ocean_thermal.process_thermal_forcing.ProcessThermalForcing`,
which handles both AIS (3D, decade-spanning files) and GrIS (2D, yearly files)
by branching on the ``ocean_3d`` parameter from ``ice_sheet_params``.

The ``run()`` method dispatches to two sub-methods based on the boolean config
options ``process_ocean_thermal`` and ``process_ocean_climatology`` in the
``[ismip7]`` section:

* ``_run_scenario()``: Processes time-varying ESM scenario data (model +
  scenario combination). Uses config from ``[ismip7_ocean_thermal]``.
* ``_run_climatology()``: Processes the static observational climatology
  (Zhou et al., AIS only). Uses config from ``[ismip7_ocean_climatology]``.

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

.. _dev_landice_ismip7_forcing_fracture:

fracture
~~~~~~~~

The :py:class:`compass.landice.tests.ismip7_forcing.fracture.Fracture`
test case processes the ISMIP7 surface-melt-driven ice shelf collapse
forcing (AIS only). It implements the three ISMIP7 pathways as independent
steps, each discovering its source file from the native
``AIS/{model}/{scenario}/fracture/{version}/`` hierarchy, building or reusing
a mapping file,
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
:py:mod:`compass.landice.ismip7.remap`:
``extrapolate_source`` (nearest-neighbor fill of NaNs on the source grid),
``open_rename_and_trim`` (open a remapped file, rename dimensions/variables
to MALI conventions, and restrict to the requested year range), and
``add_xtime_and_write`` (add the ``xtime`` variable, drop auxiliary remapping
variables, and write the output).

The output variable names for Paths A and B (``ismip7ExcessMelt``,
``ismip7LakeDepth``, ``ismip7LakeAreaFraction``) are descriptive placeholders
and may need to be aligned with the MALI Registry once the corresponding
model input fields are defined.
