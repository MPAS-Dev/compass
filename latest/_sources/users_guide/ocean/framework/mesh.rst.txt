.. _ocean_mesh:

Mesh
====

Several ocean test groups use a set of common functionality for manipulating
MPAS-Ocean meshes.  This includes remapping topography datasets to the MPAS
mesh, culling land from the mesh based on a series of masks.

.. _ocean_remap_topography:

Remapping topography
--------------------

After building a base spherical mesh (see :ref:`dev_spherical_meshes`),
the global ocean :ref:`global_ocean_mesh` includes a step for remapping
topography data (bathymetry, ocean mask, land-ice draft, land-ice thickness,
grounded and floating land-ice masks, etc.) to the MPAS mesh.  This step is
controlled by the following config options:

.. code-block:: cfg

    # config options related to remapping topography to an MPAS-Ocean mesh
    [remap_topography]

    # the name of the topography file in the bathymetry database
    topo_filename = BedMachineAntarctica_v3_and_GEBCO_2023_0.0125_degree_20230831.nc

    # variable names in topo_filename
    lon_var = lon
    lat_var = lat
    bathymetry_var = bathymetry
    ice_draft_var = ice_draft
    ice_thickness_var = thickness
    ice_frac_var = ice_mask
    grounded_ice_frac_var = grounded_mask
    ocean_frac_var = ocean_mask

    # the description to include in metadata
    description = Bathymetry is from GEBCO 2023, combined with BedMachine
                  Antarctica v3 around Antarctica.

    # the target and minimum number of MPI tasks to use in remapping
    ntasks = 1024
    min_tasks = 360

    # remapping method {'bilinear', 'neareststod', 'conserve'}
    method = conserve

The topography filename should be something from the ocean
`bathymetry database <https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/mpas-ocean/bathymetry_database/>`_.

The various variable names are designed to provide flexibility in case a given
topography file used different variable names tha the defaults provided above.

The ``description`` config option is passed on as part of the mesh metadata in
global ocean files.

The target and minimum number of MPI tasks (``ntasks`` and ``min_tasks``,
respectively) will depend on the resolution of the topography file.  The
default file is at 1/80 of a degree and typically requires at least 360 tasks
to successfully remap the data even to relatively coarse MPAS meshes.  Coarser
bathymetry datasets can likely get away with far fewer MPI tasks.  The method
used for remapping (``conserve`` by default) also makes a difference in how
many tasks are required.

Culling land cells
------------------

The framework also includes a step for culling land from the MPAS mesh,
including enforcing a series of critical passages (transects that must be
ocean, such as narrow channels) and critical land blockages (transects that
must be land, such as thin peninsulas).  The config options that can be used to
control this step are:

.. code-block:: cfg

    # options for spherical meshes
    [spherical_mesh]

    ## config options related to the step for culling land from the mesh
    # number of cores to use
    cull_mesh_cpus_per_task = 128
    # minimum of cores, below which the step fails
    cull_mesh_min_cpus_per_task = 1

To create various land masks, the culling step uses python multiprocessing.
The target and minimum number of processes are controlled by
``cull_mesh_cpus_per_task`` and ``cull_mesh_min_cpus_per_task``, respectively.
