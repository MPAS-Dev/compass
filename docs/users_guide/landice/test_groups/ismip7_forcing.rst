.. _landice_ismip7_forcing:

ismip7_forcing
==============

The ``landice/ismip7_forcing`` test group processes (i.e., remaps and renames)
the atmospheric and ocean thermal forcing data of the Ice Sheet Model
Intercomparison for CMIP7 (ISMIP7) protocol. The processed data is used to
force MALI in its simulations under the ISMIP7 experimental protocol.
The test group supports both the Antarctic Ice Sheet (AIS) and the Greenland
Ice Sheet (GrIS), controlled by a single ``ice_sheet`` config option.

The test group includes two test cases: ``atmosphere`` and ``ocean_thermal``.

* The ``atmosphere`` test case has five steps:
  ``process_smb``, ``process_temperature``, ``process_smb_gradient``,
  ``process_temperature_gradient``, and ``process_runoff``.

* The ``ocean_thermal`` test case has one step: ``process_thermal_forcing``.
  For AIS this produces 3D thermal forcing (with 30 ocean depth layers); for
  GrIS it produces 2D (depth-averaged) thermal forcing. The step can also
  process the observational ocean thermal forcing climatology (Zhou et al.)
  for AIS, controlled by the ``process_ocean_climatology`` config option.

(For more details on the steps of each test case, see
:ref:`landice_ismip7_forcing_atmosphere` and
:ref:`landice_ismip7_forcing_ocean_thermal`.)

.. _landice_ismip7_forcing_usage:

Usage
-----

To use this test group, users need to:

1. Provide a MALI mesh file onto which the source data will be remapped.

2. Set the ``ice_sheet`` config option to either ``ais`` or ``gis``.

3. Provide the path to the ISMIP7 forcing data (``base_path_ismip7``).

4. Run the ``atmosphere`` test case for each model and scenario combination.

5. Run the ``ocean_thermal`` test case for each model and scenario combination.

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
