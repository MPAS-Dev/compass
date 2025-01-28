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
    topo_filename = BedMachineAntarctica-v3_GEBCO_2023_ne3000_20250110.nc
    src_scrip_filename = ne3000_20250110.scrip.nc

    # weight generator function:
    #    `tempest` for cubed-sphere bathy or `esmf` for latlon bathy
    weight_generator = tempest

    # the description to include in metadata
    description = Bathymetry is from GEBCO 2023, combined with BedMachine
                  Antarctica v3 around Antarctica.

    # the target and minimum number of MPI tasks to use in remapping
    ntasks = 1280
    min_tasks = 256

    # remapping method {'bilinear', 'neareststod', 'conserve'}
    # must use 'conserve' for tempestremap
    method = conserve

    # threshold of what fraction of an MPAS cell must contain ocean in order to
    # perform renormalization of elevation variables
    renorm_threshold = 0.01

    # the density of land ice from MALI (kg/m^3)
    ice_density = 910.0

    # smoothing parameters
    # no smoothing (required for esmf):
    #     expandDist = 0   [m]
    #     expandFactor = 1 [cell fraction]
    expandDist = 0
    expandFactor = 1

The topography and source SCRIP filenames should be something from the ocean
`bathymetry database <https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/mpas-ocean/bathymetry_database/>`_.
The default is a ~1 km (ne3000 cubed sphere grid) dataset.

The ``weight_generator`` is the software used to generate weight files that
map between the original topography dataset and the MPAS mesh.  The ``tempest``
approach uses MOAB's ``mbtempest`` tool to generate conservative weights in
parallel.  This is the recommended approach because it is faster and more
reliable in our experience than the ``esmf`` software.  However ESMF allows
more remapping methods and supports latitude-longitude bathymetry datasets.

The ``description`` config option is passed on as part of the mesh metadata in
global ocean files.

The target and minimum number of MPI tasks (``ntasks`` and ``min_tasks``,
respectively) will depend on the resolution of the topography file.  The
default file is at ~1 km (ne3000 cubed sphere grid) and typically requires at
least 256 tasks to successfully remap the data even to relatively coarse MPAS
meshes.  Coarser bathymetry datasets can get away with far fewer MPI tasks.
The method used for remapping (``conserve`` by default) also makes a difference in how
many tasks are required.

Coarse meshes can get away with a coarser topography dataset that speeds up
the topography-remapping process. Config options related to coarser topography
are the following:

.. code-block:: cfg

    # config options related to remapping topography to an MPAS-Ocean mesh
    [remap_topography]

    # the name of the topography file in the bathymetry database
    topo_filename = BedMachineAntarctica-v3_GEBCO_2023_ne120_20250110.nc
    src_scrip_filename = ne120_20250110.scrip.nc

    # the target and minimum number of MPI tasks to use in remapping
    ntasks = 64
    min_tasks = 4

The ne120 topography dataset has a resolution of ~25 km (sufficient for MPAS
meshes with 240 km resolution) and requires far fewer processers to remap.

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
