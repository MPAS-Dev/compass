.. _dev_ocean_hurricane:

hurricane
=========

The ``hurricane`` test group implements regionally refined, single layer
barotropic, tropical cyclone cases as described in :ref:`ocean_hurricane` in
the User's Guide.

The ``hurricane`` test group uses the MPAS-Ocean wetting and drying scheme to
simulate coastal flooding due to storm surge. Three wetting and drying options
are accepted as a ``wetdry`` argument into the ``Init`` and ``Forward`` cases
(``off``, ``standard``, ``subgrid``). If ``subgrid`` is used, the test group
will build look-up tables from a high-resolution DEM in order to apply the
Shallow Water Subgrid scheme corrections to the barotropic momentum equations.

The ``hurricane`` test group also supports two local time-stepping schemes
(``LTS``, ``FB-LTS``), passed as the ``use_lts`` argument in the ``Mesh``,
``Init`` and ``Forward`` cases.

.. _dev_ocean_hurricane_meshes:

Meshes
------
The ``hurricane`` test group supports 3 meshes (``DEQU120at30cr10rr2``,
``DEVR45to5rr1``, ``RRS6to18``). Each mesh inherits from the
:py:class:`compass.ocean.mesh.floodplain.FloodplainMeshStep` class,
and provides a ``build_cell_width_lat_lon()`` method to specify the mesh
resolution array using the
:py:func:`mpas_tools.ocean.coastal_tools.coastal_refined_mesh()` function. Each
mesh also uses config options to determine how the floodplain is defined.

.. _dev_ocean_hurricane_dequ120at30cr10rr2:

DEQU120at30cr10rr2
^^^^^^^^^^^^^^^^^^
The ``DEQU120at30cr10rr2`` mesh is a quasi-uniform, 120 km horizontal resoution
global mesh with regional refinement along the Mid-Atlantic Bight down to 2 km.
The mesh is defined by
:py:class:`compass.ocean.tests.hurricane.mesh.dequ120at30cr10rr2.DEQU120at30cr10rr2BaseMesh`.

The default config options for this mesh are:

.. code-block:: cfg

    # options for spherical meshes
    [spherical_mesh]

    ## config options related to the step for culling land from the mesh
    # number of cores to use
    cull_mesh_cpus_per_task = 18
    # minimum of cores, below which the step fails
    cull_mesh_min_cpus_per_task = 1
    # maximum memory usage allowed (in MB)
    cull_mesh_max_memory = 1000

    # Elevation threshold to use for including land cells
    floodplain_elevation = 10.0


    # options for hurricane testcases
    [hurricane]

    ## config options related to the initial_state step
    # number of MPI tasks to use
    init_ntasks = 36
    # minimum of MPI tasks, below which the step fails
    init_min_tasks = 8
    # maximum memory usage allowed (in MB)
    init_max_memory = 1000
    # number of threads
    init_threads = 1

    ## config options related to the forward steps
    # number of MPI tasks to use
    forward_ntasks = 180
    # minimum of MPI tasks, below which the step fails
    forward_min_tasks = 18
    # maximum memory usage allowed (in MB)
    forward_max_memory = 1000
    # number of threads
    forward_threads = 1

.. _dev_ocean_hurricane_devr45to5rr1:

DEVR45to5rr1
^^^^^^^^^^^^
The ``DEVR45to5rr1`` mesh is a variable-resolution 45 km to 5 km global mesh
with regional refinement along the Mid-Atlantic Bight down to 1 km.
The mesh is defined by
:py:class:`compass.ocean.tests.hurricane.mesh.devr45to5rr1.DEVR45to5rr1BaseMesh`.

The default config options for this mesh are:

.. code-block:: cfg

    # options for spherical meshes
    [spherical_mesh]

    ## config options related to the step for culling land from the mesh
    # number of cores to use
    cull_mesh_cpus_per_task = 18
    # minimum of cores, below which the step fails
    cull_mesh_min_cpus_per_task = 1
    # maximum memory usage allowed (in MB)
    cull_mesh_max_memory = 1000

    # Elevation threshold to use for including land cells
    floodplain_elevation = 40.0
    # Resolution threshold to use for including land cells
    floodplain_resolution = 4.0
    # Minimum depth to enforce outside floodplain region
    min_depth_outside_floodplain = 1.0


    # options for hurricane testcases
    [hurricane]

    ## config options related to the initial_state step
    # number of MPI tasks to use
    init_ntasks = 512
    # minimum of MPI tasks, below which the step fails
    init_min_tasks = 512
    # maximum memory usage allowed (in MB)
    init_max_memory = 1000
    # number of threads
    init_threads = 1

    ## config options related to the forward steps
    # number of MPI tasks to use
    forward_ntasks = 1024
    # minimum of MPI tasks, below which the step fails
    forward_min_tasks = 1024
    # maximum memory usage allowed (in MB)
    forward_max_memory = 1000
    # number of threads
    forward_threads = 1

.. _dev_ocean_hurricane_rrs6to18:

RRS6to18
^^^^^^^^
The ``RRS6to18`` mesh is a high-resolution global mesh with Rossby-radius-scaling
of horizontal resolution from 18 km down to 6 km.
The mesh is defined by
:py:class:`compass.ocean.tests.hurricane.mesh.rrs6to18.RRS6to18BaseMesh`.

The default config options for this mesh are:

.. code-block:: cfg

    # options for spherical meshes
    [spherical_mesh]

    ## config options related to the step for culling land from the mesh
    convert_culled_mesh_to_cdf5 = True
    # number of cores to use
    cull_mesh_cpus_per_task = 18
    # minimum of cores, below which the step fails
    cull_mesh_min_cpus_per_task = 1
    # maximum memory usage allowed (in MB)
    cull_mesh_max_memory = 1000

    # Elevation threshold to use for including land cells
    floodplain_elevation = 20.0
    # Resolution threshold to use for including land cells
    floodplain_resolution = 16.0
    floodplain_geojson = mab_floodplain.geojson
    # Minimum depth to enforce outside floodplain region
    min_depth_outside_floodplain = 1.0


    # options for hurricane testcases
    [hurricane]

    ## config options related to the initial_state step
    # number of MPI tasks to use
    init_ntasks = 512
    # minimum of MPI tasks, below which the step fails
    init_min_tasks = 1
    # maximum memory usage allowed (in MB)
    init_max_memory = 1000
    # number of threads
    init_threads = 1

    ## config options related to the forward steps
    # number of MPI tasks to use
    forward_ntasks = 1024
    # minimum of MPI tasks, below which the step fails
    forward_min_tasks = 1024
    # maximum memory usage allowed (in MB)
    forward_max_memory = 1000
    # number of threads
    forward_threads = 1


Test cases
----------

.. _dev_ocean_hurricane_mesh:

mesh test case
^^^^^^^^^^^^^^
The ``mesh`` test case generates an MPAS horizontal mesh, then culls out the
land cells to improve model efficiency. For ``hurricane`` meshes, the culling
step preserves a coastal floodplain according to config options. The
:py:class:`compass.ocean.tests.hurricane.mesh.Mesh` class is instantiated with
the desired mesh name in order to build the mesh.

.. _dev_ocean_hurricane_mesh_lts:

If either of the local time-stepping schemes (LTS, FB-LTS) are enabled, the
:py:class:`compass.ocean.tests.hurricane.lts.mesh.lts_regions.LTSRegionsStep`
class creates a copy of the culled mesh file that additionally includes an
array called ``LTSRegion``.
This array has appropriate flags that determine what time-step should be used on
a certain cell of the mesh, according to the local-time stepping scheme.
The ``graph.info`` file is also copied and modified to address proper load balancing.
The aforementioned class receives the
:py:class:`compass.ocean.mesh.cull.CullMeshStep` as input.

.. _dev_ocean_hurricane_init:

init test case
^^^^^^^^^^^^^^
The ``init`` test performs steps to set up the vertical mesh, initial
conditions, atmospheric forcing, and parameterized wave and bottom drag, and
prepares the station locations for timeseries output.

initial_state
"""""""""""""
The class :py:class:`compass.ocean.tests.hurricane.init.initial_state.InitialState`
defines a step for running MPAS-Ocean in init mode. The vertical mesh is
set up with a single layer. For the subgrid scheme, the ``initial_state`` step
is where the ``wetdry`` parameter determines whether to build the subgrid
look-up tables and construct the floodplain topography averaged from the
high-resolution DEM.

interpolate_atm_forcing
"""""""""""""""""""""""
The class :py:class:`compass.ocean.tests.hurricane.init.interpolate_atm_forcing.InterpolateAtmForcing`
defines a step for interpolating CFSv2 reanalysis data for atmospheric winds
and pressure onto the MPAS-Ocean mesh at hourly time intervals. The forward
run uses this as input to update the time varying atmospheric forcing.

create_pointstats_file
""""""""""""""""""""""
The class :py:class:`compass.ocean.tests.hurricane.init.create_pointstats_file.CreatePointstatsFile`
defines a step to create the input file for the MPAS-Ocean pointWiseStats
analysis member based on station locations which have observed data.

topographic_wave_drag
"""""""""""""""""""""
The class :py:class:`compass.ocean.tests.hurricane.lts.init.topographic_wave_drag.ComputeTopographicWaveDrag`
defines a step for interpolating the reciprocal of the ``r_inv`` to the mesh edges.
This step is needed to include the contribution of the topographic wave drag
in the model momentum tendency. 

.. _dev_ocean_hurricane_sandy:

sandy test case
^^^^^^^^^^^^^^^
The sandy test case is responsible for the forward model simulation and analysis.

forward
"""""""
The class :py:class:`compass.ocean.tests.hurricane.forward.forward.ForwardStep`
defines a step to run MPAS-Ocean in forward mode. For the subgrid scheme, the
``forward`` step is where the ``wetdry`` parameter determines whether to apply
the subgrid corrections via the look-up tables built in the ``initial_state``
step.

analysis
""""""""
The class :py:class:`compass.ocean.tests.hurricane.analysis.Analysis`
defines a step to generate validation plots comparing sea surface height
timeseries between modeled and observed data at several different stations.
Both NOAA and USGS observations are plotted.
