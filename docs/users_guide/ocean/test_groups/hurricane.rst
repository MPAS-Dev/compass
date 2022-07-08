.. _ocean_hurricane:

hurricane
=========

The ``ocean/hurricane`` test group defines meshes,
initial conditions, forward simulations, and validation for global,
realistic ocean domains with regional refinement. These simulations
are forced with time-varying atmospheric reanalysis data for tropical
cyclone events and tides. The meshes contain refined regions in order
to resolve coastal estuaries, making it possible to simulate the
storm surge that results from a given hurricane event.
Currently, one mesh resolution and one storm, Hurricane Sandy, are supported.
More mesh resolutions will be supported in the future.

These test are configured to use the barotropic, single layer configuration
of MPAS-Ocean. Each mesh can optionally be created to contain the floodplain
which can be used to simulate coastal inundation using MPAS-Ocean's
wetting and drying scheme.

Shared config options
---------------------

All ``hurricane`` test cases start the following shared config options.
Note that meshes and test cases may modify these options, as noted below.

.. code-block:: cfg

    # options for global ocean testcases
    [global_ocean]

    ## config options related to the mesh step
    # number of cores to use
    mesh_ntasks = 1
    # minimum of cores, below which the step fails
    mesh_min_tasks = 1

    # The following options are detected from .gitconfig if not explicitly entered
    author = autodetect
    email = autodetect

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

.. _hurricane_mesh:

mesh test case
--------------
The mesh test case uses the mesh step from the global ocean test group.
First, it generates the global mesh based on a specified mesh resolution
function. Next, bathymetry/topography data is interpolated on the mesh from the
STRM15+ data product. This interpolation step is necessary, because the
topography in the floodplain is used to set a mask for the cell culling
process. The land cells above the ``floodplain_elevation`` are then culled
from the mesh. Finally, the bathymetry is re-interpolated onto the mesh
since this data is not carried over from the cell culling process.

.. _hurricane_init:

init test case
--------------
The init test performs steps to set up the vertical mesh, initial conditions,
atmospheric forcing, and prepares the station locations for timeseries output.

initial state step
^^^^^^^^^^^^^^^^^^
The initial state step runs MPAS-Ocean in init mode to create the initial
condition file for the forward run. The vertical mesh is setup for a
single layer case and the ssh with a thin layer on land for wetting and
drying cases.

interpolate atmosphere forcing step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The CFSv2 reanalysis wind vector components and atmospheric pressure fields
for the storm event are interpolated onto the horizontal mesh at hourly
intervals. These are read in and used to update the atmospheric forcing in the
forward run.

create pointstats file step
^^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to perform validation of the forward simulation, timeseries data
is recored at mesh cell centers which are closest to observation stations.
This set reads in the observation station locations and finds the cells
closest to them. A file is created that is the input to the
pointWiseStats analysis member for the forward run.

.. _hurricane_sandy:

sandy test case
---------------
The sandy test case is responsible for the forward model simulation and
analysis.

forward step
^^^^^^^^^^^^
The forward step runs the model simulation of the storm. The simulation
begins with a spinup period, where the tides and atmospheric forcing
are ramped to their full value to avoid shocking the system.

analysis step
^^^^^^^^^^^^^
The analysis step plots the timeseries data at each observation station
to compare the modeled and observed data. Both NOAA and USGS station data
is used for the validation.
