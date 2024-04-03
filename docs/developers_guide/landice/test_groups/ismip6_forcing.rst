.. _dev_landice_ismip6_forcing:

ismip6_forcing
==============

The ``ismip6_forcing`` test group (:py:class:`compass.landice.tests.ismip6_
forcing.Ismip6Forcing`) processes (i.e., remapping and renaming) the
atmospheric and oceanic forcing data of the Ice Sheet Model
Intercomparison for CMIP6 (ISMIP6) protocol from its native polarstereo grid to
the unstructure MALI mesh. The test group includes five test cases:
``atmosphere``, ``ocean_basal``, ``ocean_thermal_obs``, ``ocean_thermal`` and
``shelf_collapse``. The ``atmosphere`` test case has two steps: 
``process_smb`` and ``process_smb_racmo``; the ``ocean_basal`` and ``shelf_collpase``
test cases each have one step, ``process_basal_melt`` and ``process_shelf_collpase``
(respectively); the ``ocean_thermal_obs`` and ``ocean_thermal``
share one step, ``process_thermal_forcing``. Each step has the local methods
(functions) of remapping and renaming the original ISMIP6 data to the format
that MALI can incorporate in its forward simulations. In remapping the data,
all test cases import the method ``build_mapping_file`` to create or use scrip
files of the source (ISMIP6) and destination (MALI) mesh depending on the
existence of a mapping file. Below, we describe the shared framework for this
test group and the 3 test cases.

.. _dev_landice_ismip6_forcing_framework:

framework
---------

The shared config options for the ``ismip6_forcing`` test group are described
in :ref:`landice_ismip6_forcing` in the User's Guide.

create_mapfile
~~~~~~~~~~~~~~

The module :py:class:`compass.landice.tests.ismip6_forcing.create_mapfile` defines
a unified framework for creating the SCRIP and mapping files from the ISMIP6
source data files. The function 
:py:func:`compass.landice.tests.ismip6_forcing.create_mapfile.build_mapping_file`
is the common interface to build the SCRIP files and mapping files. The
``scrip_from_latlon`` keyword argument is used to call the appropriate function
for generating the SCRIP file. If ``scrip_from_latlon`` is ``false`` the
``create_SCRIP_file_from_planar_rectangular_grid.py`` command line executable
from the ``MPAS_Tools`` conda package is used, otherwise the 
:py:func:`compass.landice.tests.ismip6_forcing.create_mapfile.create_scrip_from_latlon`
function is used. Multiple methods for creating SCRIP files are necessary due to 
inconsistent dimensions names across the different ISMIP6 datasets.

Test cases
----------

.. _dev_landice_ismip6_forcing_atmosphere:

atmosphere
~~~~~~~~~~

The :py:class:`compass.landice.tests.ismip6_forcing.atmosphere.Atmosphere`
performs processing of the surface mass balance (SMB) forcing.
Processing data includes regridding the original ISMIP6 SMB data from its
native polarstereo grid to MALI's unstructured grid, renaming variables and
correcting the SMB anomaly field for the MALI base SMB.

.. _dev_landice_ismip6_forcing_ocean_basal:

ocean_basal
~~~~~~~~~~~~

The :py:class:`compass.landice.tests.ismip6_forcing.ocean_basal.OceanBasal`
performs processing of the coefficients for the basal melt parameterization
utilized by the ISMIP6 protocol. Processing data includes combining the
IMBIE2 basin number file and parameterization coefficients and remapping onto
the MALI mesh.

.. _dev_landice_ismip6_forcing_ocean_thermal:

ocean_thermal
~~~~~~~~~~~~~

The :py:class:`compass.landice.tests.ismip6_forcing.ocean_thermal.OceanThermal`
performs the processing of ocean thermal forcing, both observational climatology
(in ``ocean_thermal_obs``) and the CMIP model data (in ``ocean_thermal``).
Processing data includes regridding the original ISMIP6 thermal forcing data
from its native polarstereo grid to MALI's unstructured grid and renaming variables.

.. _dev_landice_ismip6_forcing_shelf_collapse:

shelf_collapse
~~~~~~~~~~~~~~
The :py:class:`compass.landice.tests.ismip6_forcing.shelf_collapse.ShelfCollapse`
test case performs the processing of ice shelf collapse masks by remapping the
original ISMIP6 forcing data to MALI's unstructured grid and renaming variables. 
