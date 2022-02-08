.. _config_files:

Config Files
============

``compass`` uses config files (with extension ``.cfg``) to allow users to
control how :ref:`test_cases` and :ref:`test_suites` get set up and run.

A "user" config file
--------------------

If you're running on one of the supported :ref:`machines`, and you provide a
path to where you build the MPAS model (with the ``-p`` flag to
``compass setup`` and ``compass suite``, see :ref:`setup_overview` and
:ref:`suite_overview`), you also won't need to create a config file to set up
test cases or suites.

If you're running on another machine like your own laptop, you will need to
provide some basic information for ``compass`` to work properly.  Even if
you're running on one of the supported machines, you might find it convenient
to make your own changes to config options related to either setting up or
running test suites and test case.

Here is an example:

.. code-block:: cfg

    [paths]

    mpas_model = .

    # The root to a location where the mesh_database, initial_condition_database,
    # and bathymetry_database for MPAS-Ocean will be cached
    ocean_database_root = /home/xylar/data/mpas/mpas_standalonedata/mpas-ocean

    # The root to a location where the mesh_database and initial_condition_database
    # for MALI will be cached
    landice_database_root = /home/xylar/data/mpas/mpas_standalonedata/mpas-albany-landice


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

The comments in this example are hopefully pretty self-explanatory. In this
example, the ``mpas_model`` path points to the current directory ``.``. You can
replace this with the relative or absolute path where you have built the model
(i.e. where the ``ocean_model`` or ``landice_model`` executables are found).
Since you are free to run ``compass`` from wherever you like, the example
assumes you are running from the directory where the MPAS model was built.
You may prefer to keep your ``compass`` config files in another location, in
which case you could provide an absolute path like:

.. code-block:: cfg

    # The paths section points compass to external paths
    [paths]

    # the relative or absolute path to the root of a branch where MPAS-Ocean
    # has been built
    mpas_model = /home/xylar/code/E3SM/components/mpas-ocean

You provide the config file to ``compass setup`` and ``compass suite`` with
the ``-f`` flag:

.. code-block:: bash

    compass setup -f my_machine.cfg ...

Test-case config files
----------------------

Once a test case has been set up, its work directory will contain a config file
called ``<test_case>.cfg``, where ``<test_case>`` is the name of the test case.
As a user, you can typically leave the config options in a test case as they
are to run the test in its default configuration.  But the config file is meant
to make it easier to modify the test case to fit your needs without having to
dig into the ``compass`` code.

Config options for a given test case are built up from a number of different
sources:

* the default config file,
  `default.cfg <https://github.com/MPAS-Dev/compass/blob/master/compass/default.cfg>`_,
  which sets a few options related to downloading files during setup (whether
  to download and whether to check the size of files already downloaded)

* the `machine config file <https://github.com/MPAS-Dev/compass/blob/master/compass/machines>`_
  (using `machines/default.cfg <https://github.com/MPAS-Dev/compass/blob/master/compass/machines/default.cfg>`_
  if no machine was specified) with information on the parallel system and
  the paths to cached data files

* the MPAS core's config file.  For the :ref:`ocean` core, this sets default
  paths to the MPAS-Ocean model build (including the namelist templates).  It
  uses
  `extended interpolation <https://docs.python.org/3/library/configparser.html#configparser.ExtendedInterpolation>`_
  in the config file to use config options within other config
  options, e.g. ``model = ${paths:mpas_model}/ocean_model``.

* the test group's config file if one is defined.  For idealized test groups,
  these often include the size and resolution of the mesh as well as the number
  of vertical levels.  They may include options that were flags to scripts
  or init-mode namelist options in :ref:`legacy_compass`.

* any number of config files from the test case.  There might be different
  config options depending on how the test case is configured (e.g. only if a
  certain feature is enabled.  For example, :ref:`ocean_global_ocean` loads different
  sets of config options for different meshes.

* a user's config file described above.

You are free to add any sections and config options to your config file,
in which case they will override the values specified in one of the other
config files listed above. Here is an example of some customization for the
:ref:`ocean_global_ocean` test group:

.. code-block:: cfg

    # options for global ocean testcases
    [global_ocean]

    # The following options are detected from .gitconfig if not explicitly entered
    author = Xylar Asay-Davis
    email = xylar@lanl.gov
    pull_request = https://github.com/MPAS-Dev/compass/pull/28

In this example, the author's name and email address, and the path to a pull
request will be included in the metadata for output files from this test group.

A typical config file resulting from combining all of the sources listed above
looks like:

.. code-block:: cfg

    [download]
    server_base_url = https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata
    download = True
    check_size = False
    verify = True
    core_path = mpas-ocean

    [parallel]
    partition_executable = gpmetis
    system = single_node
    parallel_executable = mpirun
    cores_per_node = 8
    threads = 8

    [paths]
    mpas_model = /home/xylar/code/mpas-work/compass/compass_1.0/E3SM-Project/components/mpas-ocean
    ocean_database_root = /home/xylar/data/mpas/mpas_standalonedata/mpas-ocean
    landice_database_root = /home/xylar/data/mpas/mpas_standalonedata/mpas-albany-landice
    baseline_dir = /home/xylar/data/mpas/test_20210413/compass_classes/ocean/global_ocean/QU240/PHC/init

    [namelists]
    forward = /home/xylar/code/mpas-work/compass/compass_1.0/E3SM-Project/components/mpas-ocean/default_inputs/namelist.ocean.forward
    init = /home/xylar/code/mpas-work/compass/compass_1.0/E3SM-Project/components/mpas-ocean/default_inputs/namelist.ocean.init

    [streams]
    forward = /home/xylar/code/mpas-work/compass/compass_1.0/E3SM-Project/components/mpas-ocean/default_inputs/streams.ocean.forward
    init = /home/xylar/code/mpas-work/compass/compass_1.0/E3SM-Project/components/mpas-ocean/default_inputs/streams.ocean.init

    [executables]
    model = /home/xylar/code/mpas-work/compass/compass_1.0/E3SM-Project/components/mpas-ocean/ocean_model

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
    init_description = Polar science center Hydrographic Climatology (PHC)
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
will need to search through this documentation (or the code on the
`compass repo <https://github.com/MPAS-Dev/compass>`_) to know what the config
options are used for.
