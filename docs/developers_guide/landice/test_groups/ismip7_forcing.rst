.. _dev_landice_ismip7_forcing:

ismip7_forcing
==============

The ``ismip7_forcing`` test group
(:py:class:`compass.landice.tests.ismip7_forcing.Ismip7Forcing`) processes
(i.e., remaps and renames) the atmospheric and ocean thermal forcing data of
the Ice Sheet Model Intercomparison for CMIP7 (ISMIP7) protocol from its
native polar stereographic grid to the MALI unstructured mesh. The test group
supports both AIS and GrIS via the ``ice_sheet`` config option. It includes
two test cases: ``atmosphere`` and ``ocean_thermal``.

.. _dev_landice_ismip7_forcing_framework:

framework
---------

The shared config options for the ``ismip7_forcing`` test group are described
in :ref:`landice_ismip7_forcing` in the User's Guide.

ice_sheet_params
~~~~~~~~~~~~~~~~

The module :py:mod:`compass.landice.tests.ismip7_forcing.ice_sheet_params`
defines a dictionary of ice-sheet-specific parameters (projection, file
naming prefix, grid resolution, data version, ocean dimensionality) and
provides the function
:py:func:`compass.landice.tests.ismip7_forcing.ice_sheet_params.get_params`
to retrieve them based on the ``ice_sheet`` config option.

configure
~~~~~~~~~

The module :py:mod:`compass.landice.tests.ismip7_forcing.configure` validates
that all required config options in the ``[ismip7]`` section have been set by
the user (i.e., are not ``NotAvailable``).

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
