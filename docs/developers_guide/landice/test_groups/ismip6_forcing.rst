.. _dev_landice_ismip6_forcing:

ismip6_forcing
==============

The ``ismip6_forcing`` test group (:py:class:`compass.landice.tests.ismip6_
forcing.Ismip6Forcing`) processes the atmospheric and oceaninc forcing data of
the Ice Sheet Model Intercomparison for CMIP6 (ISMIP6) protocol. Here,
we describe the shared framework for this test group and the 3 test cases.

.. _dev_landice_ismip6_forcing_framework:

framework
---------

The shared config options for the ``ismip6_forcing`` test group are described
in :ref:`landice_ismip6_forcing` in the User's Guide.


.. _dev_landice_ismip6_forcing_atmosphere:

atmosphere
----------

The :py:class:`compass.landice.tests.ismip6_forcing.atmosphere.Atmosphere`
performs processing of the surface mass balance (SMB) forcing.
Processing data includes regridding the original ISMIP6 SMB data from its
native polarstereo grid to MALI's unstructured grid, renaming variables and
correcting the SMB anomaly field for the MALI base SMB.

.. _dev_landice_ocean_thermal:

ocean_thermal
-------------

The :py:class:`compass.landice.tests.ismip6_forcing.ocean_thermal.OceanThermal`
performs the processing of ocean thermal forcing. Processing data includes
regridding the original ISMIP6 thermal forcing data from its native
polarstereo grid to MALI's unstructured grid and renaming variables.

.. _dev_landice_ocean_basal:

ocean_basal
------------

The :py:class:`compass.landice.tests.ismip6_forcing.ocean_basal.OceanBasal`
performs processing of the coefficients for the basal melt parametrization
utilized by the ISMIP6 protocol. Processing data includes combining the
IMBIE2 basin number file and parametrization coefficients and remapping onto
the MALI mesh.
