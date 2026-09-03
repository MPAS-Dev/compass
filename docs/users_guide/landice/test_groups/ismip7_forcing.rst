.. _landice_ismip7_forcing:

ismip7_forcing
==============

The ``landice/ismip7_forcing`` test group processes (i.e., remaps and renames)
the atmospheric and ocean thermal forcing data of the Ice Sheet Model
Intercomparison for CMIP7 (ISMIP7) protocol. The processed data is used to
force MALI in its simulations under the ISMIP7 experimental protocol.
The test group supports both the Antarctic Ice Sheet (AIS) and the Greenland
Ice Sheet (GrIS), controlled by a single ``ice_sheet`` config option.

The test group includes three test cases: ``atmosphere``, ``ocean_thermal``,
and ``fracture``.

* The ``atmosphere`` test case has five steps:
  ``process_smb``, ``process_temperature``, ``process_smb_gradient``,
  ``process_temperature_gradient``, and ``process_runoff``.

* The ``ocean_thermal`` test case has one step: ``process_thermal_forcing``.
  For AIS this produces 3D thermal forcing (with 30 ocean depth layers); for
  GrIS it produces 2D (depth-averaged) thermal forcing. The step can also
  process the observational ocean thermal forcing climatology (Zhou et al.)
  for AIS, controlled by the ``process_ocean_climatology`` config option.

* The ``fracture`` test case has three steps: ``process_excess_melt``
  (Path A), ``process_lake_properties`` (Path B), and
  ``process_shelf_collapse`` (Path C). It processes the ISMIP7
  surface-melt-driven ice shelf collapse forcing (AIS only).

(For more details on the steps of each test case, see
:ref:`landice_ismip7_forcing_atmosphere`,
:ref:`landice_ismip7_forcing_ocean_thermal`, and
:ref:`landice_ismip7_forcing_fracture`.)

.. _landice_ismip7_forcing_usage:

Usage
-----

To use this test group, users need to:

1. Provide a MALI mesh file onto which the source data will be remapped.

2. Set the ``ice_sheet`` config option to either ``ais`` or ``gis``.

3. Provide the path to the ISMIP7 forcing data (``base_path_ismip7``).

4. Run the ``atmosphere`` test case for each model and scenario combination.

5. Run the ``ocean_thermal`` test case for each model and scenario combination.

6. Run the ``fracture`` test case (AIS only) for each model and scenario
   combination to process the surface-melt-driven ice shelf collapse
   pathways (excess melt, lake properties, and the ice shelf collapse mask).

Example user config files are provided in the source tree for local testing:

* ``compass/landice/tests/ismip7_forcing/ismip7_forcing_test.cfg``
  (AIS-focused example)
* ``compass/landice/tests/ismip7_forcing/ismip7_forcing_test_gis.cfg``
  (GrIS-focused example)

These files contain machine-specific absolute paths and are intended as
templates, not portable defaults. Copy one and edit paths and options for
your environment before using it with ``compass setup ... -f USER.cfg``.

The AIS example enables both ocean processing modes and uses a 2015-2300
processing window for both atmosphere and ocean thermal scenario forcing.
The GrIS example enables scenario ocean processing only and uses 1980-2015.

For the GrIS OCX (reanalysis) scenario, use the dedicated example config
``compass/landice/tests/ismip7_forcing/ismip7_forcing_ocx_gis.cfg``. OCX has
no distinct ESM model: it uses ``RACMO2.3p2-ERA`` for the atmosphere and
``EN4`` for the ocean, both selected automatically when ``scenario = OCX``.
The ``[ismip7] model`` option is ignored for OCX (set it to ``None``), so a
single config file processes both the ``atmosphere`` and ``ocean_thermal``
test cases, just like the ESM scenarios.

.. _landice_ismip7_forcing_input_data:

Input Data
----------

ISMIP7 forcing data is organized by variable and version. The expected
directory structure under ``base_path_ismip7`` is:

For AIS atmosphere (2km, polar stereographic EPSG:3031):

.. code-block:: none

   acabf/v2/acabf_AIS_{model}_{scenario}_SDBN1-2000m_v2_{year_range}.nc
   ts/v2/ts_AIS_{model}_{scenario}_SDBN1-2000m_v2_{year_range}.nc
   dacabfdz/v2/dacabfdz_AIS_{model}_{scenario}_SDBN1-2000m_v2_{year_range}.nc
   dtsdz/v2/dtsdz_AIS_{model}_{scenario}_SDBN1-2000m_v2_{year_range}.nc
   mrro/v2/mrro_AIS_{model}_{scenario}_SDBN1-2000m_v2_{year_range}.nc

For AIS ocean thermal (8km, 30 depth levels, decade files):

.. code-block:: none

   ocean/tf/v3/tf_AIS_{model}_{scenario}_ocean_v3_{start_year}-{end_year}.nc

For AIS ocean thermal climatology (8km, 30 depth levels, static):

.. code-block:: none

   {base_path_climatology}/tf/v3/tf_AIS_obs_ocean_climatology_*.nc

For AIS fracture / ice shelf collapse mask (8km, annual, Path C):

.. code-block:: none

   fracture/v2/ice_shelf_collapse_mask_*.nc

For GrIS atmosphere (1km, polar stereographic EPSG:3413):

.. code-block:: none

   acabf/v2/acabf_GrIS_{model}_{scenario}_SDBN1-1000m_v2_{year}.nc
   ts/v2/ts_GrIS_{model}_{scenario}_SDBN1-1000m_v2_{year}.nc
   dacabfdz/v2/dacabfdz_GrIS_{model}_{scenario}_SDBN1-1000m_v2_{year}.nc
   dtsdz/v2/dtsdz_GrIS_{model}_{scenario}_SDBN1-1000m_v2_{year}.nc
   mrro/v2/mrro_GrIS_{model}_{scenario}_SDBN1-1000m_v2_{year}.nc

For GrIS ocean thermal (same 1km grid, 2D, yearly files):

.. code-block:: none

   ocean/tf/v2/tf_GrIS_{model}_{scenario}_ocean_v2_{year}.nc

The OCX (reanalysis) scenario follows the same directory layout as the ESM
scenarios, but uses fixed reanalysis sources, data version ``v1``, and a
named grid resolution in the ocean file names. For GrIS OCX the atmosphere
source is ``RACMO2.3p2-ERA`` and the ocean source is ``EN4``:

.. code-block:: none

   acabf/v1/acabf_GrIS_RACMO2.3p2-ERA_OCX_SDBN1-1000m_v1_{year}.nc
   ocean/tf/v1/tf_GrIS_EN4_OCX_ocean-1000m_v1_{year}.nc

Set ``base_path_ismip7`` to the ``OCX`` directory and ``scenario = OCX``.
The sources, version, and ocean grid token are selected automatically for
OCX, and the ``[ismip7] model`` option is ignored (set it to ``None``).

.. _landice_ismip7_forcing_config:

config options
--------------

The ``ismip7_forcing`` test group uses four config sections. The default
values are:

.. code-block:: cfg

   # config options for ismip7 forcing data
   [ismip7]

   # Ice sheet: ais (Antarctic) or gis (Greenland)
   ice_sheet = NotAvailable

   # Base path to the input ISMIP7 forcing files
   base_path_ismip7 = NotAvailable

   # Base path to the MALI mesh
   base_path_mali = NotAvailable

   # Base path to which output forcing files are saved
   output_base_path = NotAvailable

   # Name of climate model (e.g., CESM2-WACCM, MRI-ESM2-0)
   model = NotAvailable

   # Scenario (e.g., historical, ssp126, ssp370, ssp585)
   scenario = NotAvailable

   # Name of the MALI mesh (used in output file naming)
   mali_mesh_name = NotAvailable

   # MALI mesh file name
   mali_mesh_file = NotAvailable

   # Number of MPI tasks for ESMF_RegridWeightGen
   esmf_ntasks = 128

   # Whether to process time-varying ocean thermal forcing (ESM scenario data)
   process_ocean_thermal = true

   # Whether to process observational ocean thermal forcing climatology
   process_ocean_climatology = true

   # config options for ismip7 atmosphere forcing
   [ismip7_atmosphere]

   # Remapping method: bilinear, neareststod, conserve
   method_remap = conserve

   # Start year for processing
   start_year = 1850

   # End year for processing
   end_year = 2014

   # config options for ismip7 ocean thermal forcing
   [ismip7_ocean_thermal]

   # Remapping method: bilinear, neareststod, conserve
   method_remap = bilinear

   # Start year for processing
   start_year = 1850

   # End year for processing
   end_year = 2014

   # config options for ismip7 ocean thermal forcing climatology
   [ismip7_ocean_climatology]

   # Remapping method: bilinear, neareststod, conserve
   method_remap = bilinear

   # Base path to observational climatology data
   base_path_climatology = /path/to/ISMIP7/forcing/AIS/obs/zhou_annual_06_nov

   # config options for ismip7 fracture (Path C, ice shelf collapse) forcing
   [ismip7_fracture]

   # Remapping method for each pathway. Set a method to None to skip
   # processing that pathway's file.

   # Remapping method for the ice shelf collapse mask (Path C).
   # neareststod preserves the 0/1 mask values
   method_remap_shelf_collapse = neareststod

   # Remapping method for the excess meltwater field (Path A), a flux
   method_remap_excess_melt = conserve

   # Remapping method for the supraglacial lake properties (Path B)
   method_remap_lake_properties = bilinear

   # Version subdirectory of the fracture forcing data
   version = v2

   # Start year for processing
   start_year = 1850

   # End year for processing
   end_year = 2014

All ``NotAvailable`` options must be overridden in a user config file passed
at setup time (e.g., ``compass setup ... -f my_ismip7.cfg``).

The boolean options ``process_ocean_thermal`` and ``process_ocean_climatology``
control which processing paths are executed when the ``ocean_thermal`` test
case is run. Both default to ``true``. Set one to ``false`` in your user
config to skip that processing path.

.. _landice_ismip7_forcing_atmosphere:

atmosphere
----------

The ``landice/ismip7_forcing/atmosphere`` test case processes the ISMIP7
atmosphere forcing fields and remaps them from the native polar stereographic
grid to the MALI unstructured mesh.

Steps:

* **process_smb**: Remaps the surface mass balance (``acabf``) field. The
  output variable is ``sfcMassBal``.

* **process_temperature**: Remaps the surface temperature (``ts``) field,
  clipped to a maximum of 273.15 K. The output variable is
  ``surfaceAirTemperature``.

* **process_smb_gradient**: Remaps the SMB lapse rate (``dacabfdz``) field.
  The output variable is ``sfcMassBalLapseRate``.

* **process_temperature_gradient**: Remaps the temperature lapse rate
  (``dtsdz``) field. The output variable is
  ``surfaceAirTemperatureLapseRate``.

* **process_runoff**: Remaps the ice sheet runoff (``mrro``)
  field. The output variable is ``ismip6Runoff``.

.. _landice_ismip7_forcing_ocean_thermal:

ocean_thermal
-------------

The ``landice/ismip7_forcing/ocean_thermal`` test case processes the ISMIP7
ocean thermal forcing (``tf``) and remaps it from the native polar
stereographic grid to the MALI unstructured mesh.

The step supports two processing modes, controlled by boolean config options
in the ``[ismip7]`` section:

* **Scenario (time-varying) data** (``process_ocean_thermal = true``):
  Processes ESM-driven thermal forcing for a given model/scenario combination.

* **Observational climatology** (``process_ocean_climatology = true``):
  Processes the static Zhou et al. observational thermal forcing climatology
  (AIS only). This is a time-invariant 3D field referenced to 1995-2024.

Both modes can be enabled simultaneously.

For **AIS** scenario data, thermal forcing is 3D with 30 vertical ocean
layers. The input files span decades (e.g., 1850-1859). The output variable
is ``ismip6shelfMelt_3dThermalForcing`` with dimension
``nISMIP6OceanLayers``. Associated depth coordinate variables
``ismip6shelfMelt_zOcean`` and ``ismip6shelfMelt_zBndsOcean`` are also
produced.

For **AIS** climatology data, the output is the same 3D thermal forcing field
but without a Time dimension, producing a single static file.

For **GrIS**, thermal forcing is 2D (depth-averaged), with monthly temporal
resolution and yearly input files. The output variable is
``ismip6_2dThermalForcing``.

.. _landice_ismip7_forcing_fracture:

fracture
--------

The ``landice/ismip7_forcing/fracture`` test case processes the ISMIP7
surface-melt-driven ice shelf collapse forcing (AIS only). It implements the
three ISMIP7 pathways as separate steps, each remapping annual fields from
the native 8km polar stereographic grid onto the MALI unstructured mesh.

All three source files are discovered from the ``fracture/{version}/``
subdirectory of ``base_path_ismip7``.

Each pathway is run independently and can be skipped by setting its
remapping-method config option to ``None`` in the ``[ismip7_fracture]``
section (for example, ``method_remap_excess_melt = None`` skips Path A). This
is useful when only some of the pathway source files are available.

* **process_excess_melt** (Path A): Remaps the excess meltwater field
  (melt + rain after firn air content depletion), matching
  ``excess_melt_*.nc``. The output variable is ``ismip7ExcessMelt``
  (converted from mm w.e. yr-1 to SI units of kg m-2 s-1) and is written to
  ``{output_base_path}/excess_melt/{model}_{scenario}/``. Conservative
  remapping is used by default since this is a flux. This source file has no
  ``x``/``y`` coordinate variables and its array is flipped along the y axis
  relative to the other fracture files, so the step reconstructs the source
  grid (borrowing ``x``/``y`` from a sibling fracture file and flipping the
  data to match) before remapping.

* **process_lake_properties** (Path B): Remaps the supraglacial lake mean
  depth and area fraction from the Grau et al. (2025) parameterization,
  matching ``lake_properties_*.nc``. The output variables are
  ``ismip7LakeDepth`` (m) and ``ismip7LakeAreaFraction`` (unitless), written
  to ``{output_base_path}/lake_properties/{model}_{scenario}/``. Bilinear
  remapping is used by default.

* **process_shelf_collapse** (Path C): Remaps the annual ice shelf collapse
  mask, matching ``ice_shelf_collapse_mask_*.nc``.
  An ice shelf grid cell is flagged as collapsed (mask value 1) when excess
  meltwater, computed after firn air content depletion, exceeds 72.5 mm/yr for
  10 consecutive years; otherwise the mask value is 0. The mask is applied on
  floating areas only, similar to ISMIP6, and ice shelves collapse on
  January 1st of each year. The remapping uses ``neareststod`` by default so
  that the 0/1 mask values are preserved, and the remapped mask is rounded to
  0/1. The output variable is ``calvingMask`` with an accompanying ``xtime``
  variable, and the result is written to
  ``{output_base_path}/shelf_collapse/{model}_{scenario}/``.

All three pathways produce continuous fields (Paths A and B) or a discrete
mask (Path C) with an accompanying ``xtime`` variable. The output variable
names for Paths A and B (``ismip7ExcessMelt``, ``ismip7LakeDepth``,
``ismip7LakeAreaFraction``) are descriptive placeholders and may need to be
aligned with the MALI Registry once the corresponding model input fields are
defined.
