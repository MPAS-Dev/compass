.. _landice_ismip6_forcing:

ismip6_forcing
==============

The ``landice/ismip6_forcing`` test group processes (i.e., remaps and renames)
the atmospheric and ocean forcing data of the Ice Sheet Model Intercomparison for CMIP6
(ISMIP6) protocol. The processed data is used to force MALI in its simulations
under a relevant ISMIP6 (either the 2100 or 2300) experimental protocol.
The test group includes five test cases:
``atmosphere``, ``ocean_basal``, ``ocean_thermal_obs``, ``ocean_thermal`` and
``shelf_collapse``. The ``atmosphere`` test case has two steps: 
``process_smb`` and ``process_smb_racmo``; the ``ocean_basal`` and ``shelf_collpase``
test cases each have one step, ``process_basal_melt`` and ``process_shelf_collpase``
(respectively); the ``ocean_thermal_obs`` and ``ocean_thermal``
share one step, ``process_thermal_forcing``. (For more details on the steps of
each test case, see :ref:`landice_ismip6_forcing_atmosphere`,
:ref:`landice_ismip6_forcing_ocean_basal`,
:ref:`landice_ismip6_forcing_ocean_thermal_obs` and
:ref:`landice_ismip6_forcing_ocean_thermal`,
:ref:`landice_ismip6_forcing_shelf_collapse`.)
Approximated time for processing a single forcing file
on Cori (single core) is 2 and 7 minutes for the atmosphere and ocean basal
testcases, and less than a minute for ocean thermal obs and ocean thermal
testcases, respectively.

Before providing the details of necessary source data of the ISMIP6 protocols,
we provide a summary of instructions of an overall process of this test group:
To start off, users need to provide a MALI mesh onto which the source data
will be remapped and renamed. More information about obtaining these meshes will
be provided as soon as they are publicly available. Then, for a given MALI mesh,

1. run :ref:`landice_ismip6_forcing_ocean_basal` once, independent of the
model, scenario and end year.

2. run :ref:`landice_ismip6_forcing_ocean_thermal_obs` once, independent
of the model, scenario and end year, to process the thermal forcing from the observational climatology (used for
control runs).

3. run :ref:`landice_ismip6_forcing_ocean_thermal` for each model, scenario
and end year.

4. run :ref:`landice_ismip6_forcing_atmosphere` with
``process_racmo_smb = True`` once, independent of the model, scenario and
end year.


5. run :ref:`landice_ismip6_forcing_atmosphere` for each model,
scenario and end year. Users can keep ``process_racmo_smb = False`` as long as
the RACMO SMB has been processed once in Step 4, but it is harmless to leave
``process_racmo_smb = True`` as it does nothing if data is already
available, and the processing is very quick (less than a minute).

There are six different of input data sets other than the MALI mesh that
users need in order to use this package: (#1) atmospheric surface mass balance (SMB)
anomaly forcing and (#2) ocean thermal forcing for projection runs, modern
climatology files for (#3) atmospheric and (#4) ocean forcing, (#5) ISMIP6 basin
number file, and (#6) the ocean basal-melt parameter files.

Except for the file #3, The ISMIP6 source data (files #1, 2, 4-6) can be obtained by contacting the ISMIP6 steering
committee as described `here. <https://theghub.org/groups/ismip6/wiki/ISMIP6-Projections2300-Antarctica>`_
Once the users get access to the data on `Globus <https://www.globus.org>`_,
and within the base path directory that the user specifies in the config file
(i.e., the config option ``base_path_ismip6``;
see :ref:`landice_ismip6_forcing_config`), they
need to manually download the files following the directory structure
and names provided in the GHub endpoints (named ``GHub-ISMIP6-Forcing``
for the 2100 CE projection protocol and ``ISMIP6-Projections-Forcing-2300``
for the 2300 CE projection protocol). That is, the directory paths and the file
names must exactly match (even the letter case) those that are provided by the
GHub endpoints. 

.. note::
    
    It is important to emphasize that a ``base_path_ismip6`` value with a bottom 
    level directory name of ``GHub-ISMIP6-Forcing`` will **only** work for the 
    2100 CE projection protocol (``period_endyear`` config option below). The 
    same applies for a bottom level directory of ``ISMIP6-Projections-Forcing-2300``
    and the 2300 CE projection protocol. 

For example, if a user wants to process the atmosphere (SMB)
forcing representing the ``SSP585`` scenario from the ``UKESM1-0-LL`` model
provided by the ISMIP6-2100 protocol (i.e., from the ``GHub-ISMIP6-Forcing``
Globus endpoint), they must create the directory path
``AIS/Atmosphere_Forcing/UKESM1-0-LL/Regridded_8km/`` in their local system
where they will download the file ``UKESM1-0-LL_anomaly_ssp585_1995-2100_8km.nc``.
Note that the users only need to download SMB anomaly files in 8km resolution
(the climatology file is not needed) that cover the period from 1995 to
``period_endyear`` (either 2100 or 2300, defined in the config file;
see :ref:`landice_ismip6_forcing_config`).

Equivalently, for the
ocean forcing in this example, users should create the directory path
``AIS/Ocean_Forcing/ukesm1-0-ll_ssp585/1995-2100/`` and download the file
``UKESM1-0-LL_ssp585_thermal_forcing_8km_x_60m.nc`` from the same endpoint.
Users do not need to download the thermal
forcing files for the years previous to 1995 as only the files downloaded from
``1995-{period_endyear}`` will be processed. Users also do not need to download
the temperature and salinity files, as these will not be used by MALI.
Also note to be aware that unlike those in the ``GHub-ISMIP6-Forcing`` endpoint,
the directory names in the ``ISMIP6-Projections-Forcing-2300`` endpoint have a
lower case "f" for the ``AIS/Atmospheric_forcing/`` and ``AIS/Ocean_forcing/``.

In addition to atmospheric and ocean thermal forcing files that
correspond to specific climate model (e.g., UKESM1-0-LL, CCSM4) and scenarios
(e.g., SSP585, RCP85, RCP26-repeat), modern
climatology files are needed. For the ``atmosphere`` testcase,
``RACMO2.3p2_ANT27_smb_yearly_1979_2018.nc`` will be automatically downloaded
from the MALI public database when the testcase is being set up and saved
to a subdirectory of the root directory that users define in the config option
``database_root`` (defined automatically on supported machines).
The RACMO file is used to correct the ISMIP6 the surface mass balance (SMB)
data with the modern climatology. For the ``ocean_thermal`` case, users need to
download the modern ocean thermal forcing climatology file named
``obs_thermal_forcing_1995-2017_8km_x_60m.nc`` in the directory
``AIS/Ocean_F{f}orcing/climatology_from_obs_1995-2017/``
(the salinity and temperature files do not have to be downloaded).


For the ``ocean_basal`` testcase, users need to additionally download
the basin number file ``imbie2_basin_numbers_8km.nc`` in the directory
``AIS/Ocean_Forcing/imbie2/`` (or ``AIS/Ocean_forcing/imbie2/``, if from the
``ISMIP6-Projections-Forcing-2300`` endpoint); all of the files that
start their name with ``coeff_gamma0_DeltaT_quadratic_local`` in the directory
''AIS/Ocean_F{f}orcing/parameterizations/'', which contain parameter values needed
for calculating the basal melt underneath the ice shelves in MALI simulations.

Note that both the RACMO SMB data and ocean basal-melt parameters not
associated with any climate models and scenarios and thus can be processed only
once and can be applied to MALI with any set of processed climate forcing data.


In the next section (ref:`landice_ismip6_forcing_config`), we provide
instructions and examples of how users can configure necessary options including
paths to necessary source files and the output path of the processed data
within which the subdirectories called ``atmosphere_forcing/``, ``basal_melt/``
and ``ocean_thermal_forcing/`` (and further subdirectories that match the source
file directory structure) are created if the directories do not already exist)
and where processed files will be saved.

.. _landice_ismip6_forcing_config:

config options
--------------

All five test cases share some set of default config options under the section
``[ismip6_ais]`` and have separate config options for each test case:
``[ismip6_ais_atmosphere]``, ``[ismip6_ais_ocean_thermal]``, 
``[ismip6_ais_ocean_basal]``, and ``[ismip6_ais_shelf_collapse``]. In the
general config section (``[ismip6_ais]``), users need to supply base paths to
input files and MALI mesh file, and MALI mesh name, as well as the model name,
climate forcing scenario and the projection end year of the ISMIP6 forcing data,
which can be chosen from the available options as given in the config file
(see the example file below). In the ``ismip6_ais_atmosphere`` section,
users need to indicate ``True`` or ``False`` on whether to process the RACMO
modern climatology (``True`` is required to run the ``process_smb_racmo`` step,
which needs to be run before the ``process_smb`` step).

The ``[ismip6_ais_atmosphere]`` and ``[ismip6_ais_ocean_thermal]``
config sections allow users to choose the interpolation scheme among
``bilinear``, ``neareststod`` and ``conserve`` methods. The exception is that
the ``ocean basal`` test case will always use the ``neareststod`` method
because the source files have a single valued data per basin. Futhermore, the 
``[ismip6_ais_atmosphere]`` and ``[ismip6_ais_shelf_collpase]`` config sections
support a ``data_resolution`` config option, which allows the user to pick the
source data resolution most appropriate for the MALI mesh the data is being 
interpolated onto. The ``[ismip6_ais_ocean_thermal]`` config section does not 
support the ``data_resolution`` config option, because the source datasets are
only provided at a single resolution.

Below are the default config options:

.. code-block:: cfg

    # config options for ismip6 antarctic ice sheet data set
    [paths]
    # The root to a location where data files for MALI will be cached
    database_root = /Users/hollyhan/Desktop/RESEARCH/MALI/database/

    [ismip6_ais]

    # Base path to the input ismip6 ocean and smb forcing files. User has to supply.
    base_path_ismip6 = /Users/hollyhan/Desktop/ISMIP6_2300_Protocol/ISMIP6-Projections-Forcing-2300/

    # Base path to the the MALI mesh. User has to supply.
    base_path_mali = /Users/hollyhan/Desktop/RESEARCH/MALI/mesh_files/

    # Forcing end year of the ISMIP6 data. User has to supply.
    # Available end years are 2100 and 2300.
    period_endyear = 2300

    # Base path to which output forcing files are saved.
    output_base_path = /Users/hollyhan/Desktop/ISMIP6_2300_Protocol/Process_Forcing_Testcase/

    # Name of climate model name used to generate ISMIP6 forcing data. User has to supply.
    # Available model names for the 2100 projection are the following: CCSM4, CESM2, CNRM_CM6, CNRM_ESM2, CSIRO-Mk3-6-0, HadGEM2-ES, IPSL-CM5A-MR, MIROC-ESM-CHEM, NorESM1-M, UKESM1-0-LL
    # Available model names for the 2300 projection are the following: CCSM4, CESM2-WACCM, CSIRO-Mk3-6-0, HadGEM2-ES, NorESM1-M, UKESM1-0-LL
    model = NorESM1-M

    # Scenarios used by climate model. User has to supply.
    # Available scenarios for the 2100 projection are the following: RCP26, RCP26-repeat, RCP85, SSP126, SSP585 (SSP585v1 and SSP585v2 for the CESM2 model)
    # Available scenarios for the 2300 projection are the following: RCP26, RCP26-repeat, RCP85, RCP85-repeat, SSP126, SSP585, SSP585-repeat
    scenario = RCP26-repeat

    # name of the mali mesh. User has to supply. Note: It is used to name mapping files
    # (e,g. 'map_ismip6_8km_to_{mali_mesh_name}_{method_remap}.nc').
    mali_mesh_name = Antarctica_8to30km

    # MALI mesh file to be used to build mapping file (e.g.Antarctic_8to80km_20220407.nc). User has to supply.
    mali_mesh_file = AIS_8to30km_r01_20220607.nc

    # config options for ismip6 antarctic ice sheet SMB forcing data test cases
    [ismip6_ais_atmosphere]
    
    # resolution of CMIP6 model data to be used; supported options are [8km, 4km]
    data_resolution = 8km

    # Remapping method used in building a mapping file. Options include: bilinear, neareststod, conserve
    method_remap = bilinear

    # Set True to process RACMO modern climatology
    process_smb_racmo = True

    # config options for ismip6 antarctic ice shelf collpase forcing test cases
    [ismip6_ais_shelf_collapse]

    # resolution of CMIP6 model data to be used; supported options are [8km, 4km]
    data_resolution = 8km

    # config options for ismip6 ocean thermal forcing data test cases
    [ismip6_ais_ocean_thermal]

    # Remapping method used in building a mapping file. Options include: bilinear, neareststod, conserve
    method_remap = bilinear

    # Set to True if the want to process observational thermal forcing data. Set to False if want to process model thermal forcing data.
    process_obs_data = True

Below is the example config options that users might create in running
the test group. This example is for processing the NorESM1-M RCP2.6 repeat
forcing to the year 2300 onto the 8-80km Antarctic Ice Sheet MALI mesh.
The example is configured to perform the `atmosphere\process_smb_racmo` step to
process the RACMO modern SMB climatology but not the modern thermal forcing.

.. code-block:: cfg

    # config options for ismip6 antarctic ice sheet data set
    [paths]
    # The root to a location where data files for MALI will be cached
    database_root = NotAvailable

    [ismip6_ais]

    # Base path to the input ismip6 ocean and smb forcing files. User has to supply.
    base_path_ismip6 = NotAvailable

    # Base path to the the MALI mesh. User has to supply.
    base_path_mali = NotAvailable

    # Forcing end year of the ISMIP6 data. User has to supply.
    # Available end years are 2100 and 2300.
    period_endyear = NotAvailable

    # Base path to which output forcing files are saved.
    output_base_path = NotAvailable

    # Name of climate model name used to generate ISMIP6 forcing data. User has to supply.
    # Available model names for the 2100 projection are the following: CCSM4, CESM2, CNRM_CM6, CNRM_ESM2, CSIRO-Mk3-6-0, HadGEM2-ES, IPSL-CM5A-MR, MIROC-ESM-CHEM, NorESM1-M, UKESM1-0-LL
    # Available model names for the 2300 projection are the following: CCSM4, CESM2-WACCM, CSIRO-Mk3-6-0, HadGEM2-ES, NorESM1-M, UKESM1-0-LL
    model = NotAvailable

    # Scenarios used by climate model. User has to supply.
    # Available scenarios for the 2100 projection are the following: RCP26, RCP26-repeat, RCP85, SSP126, SSP585 (SSP585v1 and SSP585v2 for the CESM2 model)
    # Available scenarios for the 2300 projection are the following: RCP26, RCP26-repeat, RCP85, RCP85-repeat, SSP126, SSP585, SSP585-repeat
    scenario = NotAvailable

    # name of the mali mesh. User has to supply. Note: It is used to name mapping files
    # (e,g. 'map_ismip6_8km_to_{mali_mesh_name}_{method_remap}.nc').
    mali_mesh_name = NotAvailable

    # MALI mesh file to be used to build mapping file (e.g.Antarctic_8to80km_20220407.nc). User has to supply.
    mali_mesh_file = NotAvailable

    # config options for ismip6 antarctic ice sheet SMB forcing data test cases
    [ismip6_ais_atmosphere]

    # resolution of CMIP6 model data to be used; supported options are [8km, 4km]
    data_resolution = 8km

    # Remapping method used in building a mapping file. Options include: bilinear, neareststod, conserve
    method_remap = bilinear

    # Set True to process RACMO modern climatology
    process_smb_racmo = True

    # config options for ismip6 antarctic ice shelf collpase forcing test cases
    [ismip6_ais_shelf_collapse]

    # resolution of CMIP6 model data to be used; supported options are [8km, 4km]
    data_resolution = 8km

    # config options for ismip6 ocean thermal forcing data test cases
    [ismip6_ais_ocean_thermal]

    # Remapping method used in building a mapping file. Options include: bilinear, neareststod, conserve
    method_remap = bilinear

    # Set to True if the want to process observational thermal forcing data. Set to False if want to process model thermal forcing data.
    process_obs_data = True

.. _landice_ismip6_forcing_atmosphere:

atmosphere
----------

The ``landice/ismip6_forcing/atmosphere`` test case
performs processing of the surface mass balance (SMB) forcing data provided by
the ISMIP6 and RACMO. Processing data includes regridding the SMB forcing data
SMB data from the native grid (polarstereo grid for the ISMIP6 files and
rotated pole grid for the RACMO file) to MALI's unstructured grid, renaming
variables, and correcting the ISMIP6 SMB anomaly field for the base SMB
(modern climatology) provided by RACMO.

.. _landice_ismip6_forcing_ocean_basal:

ocean_basal
------------

The ``landice/tests/ismip6_forcing/ocean_basal`` test case
performs processing of the coefficients for the basal melt parameterization
utilized by the ISMIP6 protocol. Processing data includes combining the
IMBIE2 basin numbers file and parameterization coefficients and remapping onto
the MALI mesh.

.. _landice_ismip6_forcing_ocean_thermal_obs:

ocean_thermal_obs
-----------------

The ``landice/ismip6_forcing/ocean_thermal_obs`` test case
performs the processing of the observational climatology of
ocean thermal forcing. Processing data includes regridding the original ISMIP6
thermal forcing data from its native polarstereo grid to MALI's unstructured
grid and renaming variables.

.. _landice_ismip6_forcing_ocean_thermal:

ocean_thermal
-------------

The ``landice/ismip6_forcing/ocean_thermal`` test case
performs the processing of ocean thermal forcing. Processing data includes
regridding the original ISMIP6 thermal forcing data from its native
polarstereo grid to MALI's unstructured grid and renaming variables.

.. _landice_ismip6_forcing_shelf_collapse:

shelf_collapse
--------------
The ``landice/ismip6_forcing/shelf_collpase`` test case performs the processing
of ice shelf collapse masks by remapping the original ISMIP6 forcing data to
MALI's unstructured grid and renaming variables. This test case is only supported
with a ``period_endyear`` of ``2300``.
