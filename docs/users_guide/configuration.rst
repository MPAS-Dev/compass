.. _config_files:

Configuration Files
===================

compass uses configuration files (with extension ``.cfg``) to allow users to
control how :ref:`test_cases` and :ref:`test_suites` get set up and run.
Configuration options for a given test case are built up from a number of
different sources:

* the default config file,
  `default.cfg <https://github.com/MPAS-Dev/compass/blob/master/compass/default.cfg>`_,
  which sets a few options related to downloading files during setup (whether
  to download and whether to check the size of files already downloaded)

* the `machine config file <https://github.com/MPAS-Dev/compass/blob/master/compass/machines>`_
  (using `machines/default.cfg <https://github.com/MPAS-Dev/compass/blob/master/compass/machines/default.cfg>`_
  if no machine was specified) with information on the parallel system and
  the paths to cached data files

* the core's config file.  For the :ref:`ocean` core, this sets default paths
  to the MPAS-Ocean model build (including the namelist templates).  It uses
  `extended interpolation <https://docs.python.org/3/library/configparser.html#configparser.ExtendedInterpolation>`_
  in the config file to use config options within other config
  options, e.g. ``model = ${paths:mpas_model}/ocean_model``.

* the configuration's config file if one is defined.  For idealized
  configurations, these include config options that were init-mode namelist
  options in :ref:`legacy_compass`.  For :ref:`ocean_global_ocean`, these include
  defaults for mesh metadata (again using
  `extended interpolation <https://docs.python.org/3/library/configparser.html#configparser.ExtendedInterpolation>`_);
  the default number of cores and other resource usage for mesh, init and
  forward steps; and options related to files created for E3SM initial
  conditions.

* any number of config files from the test case.  There might be different
  config options depending on how the test case is configured (e.g. only if a
  certain feature is enabled.  For example, :ref:`ocean_global_ocean` loads different
  sets of config options for different meshes.

* a user's config file that is passed in to the ``compass setup`` (see
  :ref:`setup_overview`) or ``compass suite`` (see :ref:`suite_overview`).

As a user, this last one is the only config file you will need to set up
yourself.  If you are on one of the :ref:`supported_machines` and you have
built the MPAS-Model code in the default location (see :ref:`compass_repo`),
you won't need to create a config file at all.

If you are using a machine that ``compass`` doesn't know about, you will need
to provide some config options to tell it where you would like to cache some
data files and how many cores you have:

.. code-block:: cfg

    # The paths section points compass to external paths
    [paths]

    # The mesh_database and the initial_condition_database are locations where
    # meshes / initial conditions might be found on a specific machine. They can be
    # the same directory, or different directory. Additionally, if they are empty
    # some test cases might download data into them, which will then be reused if
    # the test case is run again later.
    mesh_database = /home/xylar/data/mpas/meshes
    initial_condition_database = /home/xylar/data/mpas/initial_conditions
    bathymetry_database = /home/xylar/data/mpas/bathymetry_database


    # The parallel section describes options related to running tests in parallel
    [parallel]

    # parallel system of execution: slurm or single_node
    system = single_node

    # whether to use mpirun or srun to run the model
    parallel_executable = mpirun

    # cores per node on the machine
    cores_per_node = 8

    # the number of multiprocessing or dask threads to use
    threads = 8

The comments in this example are hopefully pretty self-explanatory.  If you
have build MPAS-Model in another location, which is often useful if you are
developing the model yourself, you will need to provide the path to the
location where the model is build (i.e. where the ``ocean_model`` or
``landice_model`` executables are found):

.. code-block:: cfg

    # The paths section points compass to external paths
    [paths]

    # the relative or absolute path to the root of a branch where MPAS-Ocean
    # has been built
    mpas_model = /home/xylar/code/MPAS-Model/ocean/develop

You are free to add any other sections and config options to your config file,
in which case they will override the values specified in one of the other
config files listed above.

A typical config file resulting from all of this looks like:

.. code-block:: cfg

    [download]
    download = True
    check_size = False
    verify = True

    [parallel]
    system = single_node
    parallel_executable = mpirun
    cores_per_node = 8
    threads = 8

    [paths]
    mpas_model = /home/xylar/code/mpas-work/compass/compass_1.0/MPAS-Model/ocean/develop
    mesh_database = /home/xylar/data/mpas/meshes
    initial_condition_database = /home/xylar/data/mpas/initial_conditions
    bathymetry_database = /home/xylar/data/mpas/bathymetry_database

    [namelists]
    forward = /home/xylar/code/mpas-work/compass/compass_1.0/MPAS-Model/ocean/develop/default_inputs/namelist.ocean.forward
    init = /home/xylar/code/mpas-work/compass/compass_1.0/MPAS-Model/ocean/develop/default_inputs/namelist.ocean.init

    [streams]
    forward = /home/xylar/code/mpas-work/compass/compass_1.0/MPAS-Model/ocean/develop/default_inputs/streams.ocean.forward
    init = /home/xylar/code/mpas-work/compass/compass_1.0/MPAS-Model/ocean/develop/default_inputs/streams.ocean.init

    [executables]
    model = /home/xylar/code/mpas-work/compass/compass_1.0/MPAS-Model/ocean/develop/ocean_model

    [ssh_adjustment]
    iterations = 10

    [global_ocean]
    mesh_cores = 1
    mesh_min_cores = 1
    mesh_max_memory = 1000
    mesh_max_disk = 1000
    init_cores = 4
    init_min_cores = 1
    init_max_memory = 1000
    init_max_disk = 1000
    init_threads = 1
    forward_cores = 4
    forward_min_cores = 1
    forward_threads = 1
    forward_max_memory = 1000
    forward_max_disk = 1000
    add_metadata = True
    prefix = QU
    mesh_description = MPAS quasi-uniform mesh for E3SM version ${e3sm_version} at
        ${min_res}-km global resolution with ${levels} vertical
        level
    bathy_description = Bathymetry is from GEBCO 2019, combined with BedMachine Antarctica around Antarctica.
    init_description = <<<Missing>>>
    e3sm_version = 2
    mesh_revision = 1
    min_res = 240
    max_res = 240
    max_depth = autodetect
    levels = autodetect
    creation_date = autodetect
    author = Xylar Asay-Davis
    email = xylar@lanl.gov
    pull_request = https://github.com/MPAS-Dev/compass/pull/28

    [files_for_e3sm]
    enable_ocean_initial_condition = true
    enable_ocean_graph_partition = true
    enable_seaice_initial_condition = true
    enable_scrip = true
    enable_diagnostics_files = true
    comparisonlatresolution = 0.5
    comparisonlonresolution = 0.5
    comparisonantarcticstereowidth = 6000.
    comparisonantarcticstereoresolution = 10.
    comparisonarcticstereowidth = 6000.
    comparisonarcticstereoresolution = 10.

    [vertical_grid]
    grid_type = tanh_dz
    vert_levels = 16
    bottom_depth = 3000.0
    min_layer_thickness = 3.0
    max_layer_thickness = 500.0

Unfortunately, all comments are lost in the process of combining config
options.  Comments are not parsed by ``ConfigParser``, and there is not a
standard for which comments are associated with which options.  So users
will need to search through this documentation to know what the config options
are used for.
