.. _landice_ismip6_GrIS_forcing:

ismip6_GrIS_forcing
===================

The ``landice/ismip6_GrIS_forcing`` test group processes (i.e., remaps and
renames) the atmospheric and ocean forcing data for Greenland Ice Sheet (GrIS)
projections, following the ISMIP6 
`protocol <https://theghub.org/groups/ismip6/wiki/ISMIP6-Projections-Greenland>`_
. The test group includes a single test case, ``forcing_gen``, which has three
steps: ``create_mapping_files``, ``smb_ref_climatology``, and
``process_forcing``.

The test group supports the following experiments, as defined in the 
``experiments.yml`` file:

- ``ctrl``: control run using RACMO atmosphere and MIROC5 ocean forcing,
  averaged over the respective reference climatology periods
- ``hist``: historical run using RACMO atmosphere and MIROC5 ocean forcing
- ``Exp05``: MIROC5 RCP8.5 projection (2015–2100)
- ``Exp06``: NorESM1-M RCP8.5 projection (2015–2100)
- ``Exp07``: MIROC5 RCP2.6 projection (2015–2100)
- ``Exp08``: HadGEM2-ES RCP8.5 projection (2015–2100)

Atmospheric and ocean forcing fields are provided by ISMIP6 on a
regularly spaced north polar stereographic ( `EPSG:3413 <https://epsg.io/3413>`_ )
grid. We use bilinear interpolation a remap the forcing data from the regularly
spaced grid onto the unstructed MALI mesh.

If running on a machine other than perlmutter, users must provide:

1. The ISMIP6 GrIS forcing archive (available via the ISMIP6 GHub Globus
   endpoint ``GHub-ISMIP6-Forcing``). Files must be stored in their native
   directory structure under a ``GrIS/`` top-level subdirectory, as provided
   by the archive.
2. RACMO2.3 SMB and grid files for computing the SMB reference climatology.

.. _landice_ismip6_GrIS_forcing_config:

config options
--------------

The ``forcing_gen`` test case uses three config sections:
``[ISMIP6_GrIS_Forcing]``, ``[smb_ref_climatology]``, and
``[TF_ref_climatology]``. Users must supply a value for ``MALI_mesh_fp``.
All other options have defaults appropriate for supported
machines (i.e. perlmutter).

Below are the default config options:

.. code-block:: cfg

    [smb_ref_climatology]

    # Path to directory containing RACMO SMB and grid data
    racmo_directory = /global/cfs/cdirs/fanssie/standard_datasets/RACMO2p3

    # Filename of the RACMO grid descriptor file
    racmo_grid_fn = Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS.nc

    # Filename of the RACMO monthly SMB record
    racmo_smb_fn = smb_rec.1958-2019.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc

    # Start year for computing the SMB reference climatology
    climatology_start = 1960

    # End year for computing the SMB reference climatology
    climatology_end = 1989

    [TF_ref_climatology]

    # Start year of the ocean thermal forcing reference period (used for ctrl)
    climatology_start = 1990

    # End year of the ocean thermal forcing reference period (used for ctrl)
    climatology_end = 2014

    [ISMIP6_GrIS_Forcing]

    # Full path to the MALI mesh file that forcing data will be regridded onto.
    # User must supply.
    MALI_mesh_fp = /path/to/mali_mesh.nc

    # Path to the GrIS subdirectory of the ISMIP6-Forcing-Ghub archive.
    # User must supply.
    archive_fp = /global/cfs/cdirs/fanssie/standard_datasets/ISMIP6-Forcing-Ghub/GrIS

    # List of experiments to generate forcing for.
    # Supported experiments: ctrl, hist, Exp05, Exp06, Exp07, Exp08
    # (see experiments.yml packaged with the test group for full definitions)
    experiments = ctrl, hist, Exp05, Exp06, Exp07, Exp08

.. _landice_ismip6_GrIS_forcing_gen:

forcing_gen
-----------

The ``landice/ismip6_GrIS_forcing/forcing_gen`` test case remaps and processes
ISMIP6 GrIS forcing data onto the user-specified MALI mesh. It consists of
three steps run sequentially.

.. _landice_ismip6_GrIS_forcing_create_mapping_files:

create_mapping_files
~~~~~~~~~~~~~~~~~~~~

This step generates the SCRIP grid descriptor files and bilinear mapping
weight files needed to remap forcing data to the MALI mesh. Two sets of
mapping files are produced:

- **RACMO to MALI**: for remapping the RACMO2.3 SMB reference climatology
- **ISMIP6 GrIS to MALI**: for remapping all ISMIP6 projection forcing
  variables

All forcing files in the ISMIP6 GrIS archive are on the same polar
stereographic grid, so only one ISMIP6 mapping file is needed regardless of
the number of experiments requested.

The mapping files are written to the ``mapping_files/`` subdirectory of the
test case work directory and are shared by all subsequent steps. This step
uses up to 128 CPUs (via ``srun``) to run ``ESMF_RegridWeightGen``.

.. _landice_ismip6_GrIS_forcing_smb_ref_climatology:

smb_ref_climatology
~~~~~~~~~~~~~~~~~~~

This step remaps the RACMO monthly SMB record onto the MALI mesh and computes
a time-mean climatology over the period defined by ``climatology_start`` and
``climatology_end`` in the ``[smb_ref_climatology]`` config section. The
resulting ``sfcMassBal`` climatology (in units of
kg m\ :sup:`-2` s\ :sup:`-1`) is saved to
``racmo_climatology_{start}--{end}.nc`` and is used in the ``process_forcing``
step to convert ISMIP6 SMB anomalies to full SMB fields.

.. _landice_ismip6_GrIS_forcing_process_forcing:

process_forcing
~~~~~~~~~~~~~~~

This step loops over the requested experiments and produces two output forcing
files for each experiment:

- ``gis_atm_forcing_{GCM}_{scenario}_{start}--{end}.nc`` contains the
  atmospheric forcing variables:

  - ``sfcMassBal``
  - ``sfcMassBal_lapseRate``
  - ``surfaceAirTemperature``
  - ``surfaceAirTemperature_lapseRate``

- ``gis_ocn_forcing_{GCM}_{scenario}_{start}--{end}.nc`` contains the ocean
  forcing variables:

  - ``ismip6_2dThermalForcing``
  - ``ismip6Runoff``

The atmospheric SMB (``sfcMassBal``) and surface air temperature
(``surfaceAirTemperature``) fields are provided as anomalies by ISMIP6. The
step adds the RACMO SMB reference climatology (from the ``smb_ref_climatology``
step) and the MALI mesh surface air temperature baseline to those anomalies,
respectively, to produce full forcing fields. Ocean variables (thermal forcing
and runoff) have NaN values (e.g., above sea level or outside
marine-terminating basins) replaced with zero.

For the ``ctrl`` experiment, the atmosphere uses zeroed RACMO anomalies and the
ocean uses MIROC5 RCP8.5, with each field time-averaged over the relevant
reference climatology period. For the ``hist`` experiment, RACMO and MIROC5
RCP8.5 data are used for the 1989–2015 period. For projection experiments
(``Exp05`` through ``Exp08``), GCM-specific forcing is used for 2015–2100.
