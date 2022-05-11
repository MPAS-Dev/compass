.. _landice_ismip6_forcing:

hydro_radial
============

The ``landice/ismip6_forcing`` test group implements processing of atmospheric
and ocean forcing data of the Ice Sheet Model Intercomparison for CMIP6
(ISMIP6) protocol.

The test group includes 3 test cases, ``atmosphere``, ``ocean_thermal`` and
``ocean_basal``. All test cases are made up of a single main step,
``ProcessSMB``,``ProcessThermalForcing`` and ``ProcessBasalMelt``,
respectively. Each step has the local methods (functions) of remapping and
renaming the original ismip6 data to the format that MALI can incorporate in
its forward simulations. In remapping the data, all test cases import the
method ``build_mapping_file`` to create or use scrip files
of the source (ISMIP6) and destination (MALI) mesh depending on the existence
of a mapping file.

config options
--------------

All three test cases share some set of default config options under the section
``[ismip6_ais]`` and separate config options under the corresponding section:

.. code-block:: cfg

    # config options for ismip6 antarctic ice sheet SMB forcing data set
    [ismip6_ais]

    # Base path to the input ismip6 ocean and smb forcing files. User has to supply.
    base_path_ismip6 = /Users/hollyhan/Desktop/Data/ISMIP6-Projections-Forcing-2300/

    # Base path to the the MALI mesh. User has to supply.
    base_path_mali = /Users/hollyhan/Desktop/Data/MALI-mesh/

    # Forcing end year of the ISMIP6 data. User has to supply.
    # Available end years are 2100 and 2300.
    period_endyear = 2300

    # Name of climate model name used to generate ISMIP6 forcing data. User has to supply.
    # Available model names are the following: CCSM4, CESM2-WACCM, CSIRO-Mk3-6-0, HadGEM2-ES, NorESM1-M, UKESM1-0-LL
    model = NorESM1-M

    # Scenarios used by climate model. User has to supply.
    # Available scenarios are the following: RCP26, RCP85, SSP126, SSP585
    scenario = RCP26-repeat

    # name of the mali mesh used to name mapping files (e,g. Antarctica_8to80km). User has to supply.
    mali_mesh_name = Antarctica_8to80km

    # MALI mesh file to be used to build mapping file (netCDF format). User has to supply.
    mali_mesh_file = Antarctica_8to80km_20220407.nc


    # config options for ismip6 antarctic ice sheet SMB forcing data test cases
    [ismip6_ais_atmosphere]

    # Remapping method used in building a mapping file. Options include: bilinear, neareststod, conserve
    method_remap = bilinear

    # Path to which processed output SMB data is saved. User has to supply.
    output_path = /Users/hollyhan/Desktop/Data/ISMIP6-Projections-Forcing-2300/AIS_Processed/Atmospheric_Forcing/


    # config options for ismip6 ocean thermal forcing data test cases
    [ismip6_ais_ocean_thermal]

    # Remapping method used in building a mapping file. Options include: bilinear, neareststod, conserve
    method_remap = bilinear

    # Path to which processed output thermal forcing data is saved. User has to supply.
    output_path = /Users/hollyhan/Desktop/Data/ISMIP6-Projections-Forcing-2300/AIS_Processed/Ocean_Forcing/


    # config options for ismip6 ocean basal test case
    [ismip6_ais_ocean_basal]

    # Remapping method used in building a mapping file. Options include: bilinear, neareststod, conserve
    method_remap = neareststod

    # Path to which processed output basal param coeff data is saved. User has to supply.
    output_path = /Users/hollyhan/Desktop/Data/ISMIP6-Projections-Forcing-2300/AIS_Processed/Ocean_Forcing/parametrizations/


atmosphere
----------

The ``landice/ismip6_forcing/atmosphere``
performs processing of the surface mass balance (SMB) forcing.
Processing data includes regridding the original ISMIP6 SMB data from its
native polarstereo grid to MALI's unstructured grid, renaming variables and
correcting the SMB anomaly field for the MALI base SMB.

ocean_thermal
-------------

The ``landice/ismip6_forcing/ocean_thermal``
performs the processing of ocean thermal forcing. Processing data includes
regridding the original ISMIP6 thermal forcing data from its native
polarstereo grid to MALI's unstructured grid and renaming variables.

ocean_basal
------------

The ``landice.tests.ismip6_forcing.ocean_basal``
performs processing of the coefficients for the basal melt parametrization
utilized by the ISMIP6 protocol. Processing data includes combining the
IMBIE2 basin number file and parametrization coefficients and remapping onto
the MALI mesh.

