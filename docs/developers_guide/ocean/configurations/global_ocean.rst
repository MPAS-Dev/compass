.. _dev_ocean_global_ocean:

global_ocean
============

The ``global_ocean`` configuration creates meshes and initial conditions, and
performs testing and dynamic adjustment for global, realistic ocean domains.
It includes 9 test cases on 4 meshes, with the expectation that more meshes
from :ref:`legacy_compass` will be added in the near future.

.. _dev_ocean_global_ocean_framework:

Framework
---------

The shared configuration options for the ``global_ocean`` configuration
are described in :ref:`ocean_global_ocean` in the User's Guide.

Additionally, the configuration has several shared namelist and streams files,
some for shared parameters and streams for forward runs (``namelist.forward``
and ``streams.forward``), one specific to meshes with ice-shelf cavities
(``namelist.wisc``), and some related to simulations with biogeochemistry
(``namelist.bgc`` and ``streams.bgc``).

.. _dev_ocean_global_ocean_description:

description
~~~~~~~~~~~

The module ``compass.ocean.tests.global_ocean.description`` creates a
description of a test case (basically the long name of the testcase) given its
parameters.  The name includes the name of the mesh, the initial condition and
whether biogeochemistory is included, the time integrator and a description
specific to the test case itself.

.. _dev_ocean_global_ocean_meshes:

meshes
~~~~~~

The module ``compass.ocean.tests.global_ocean.mesh`` defines the
:ref:`dev_ocean_global_ocean_mesh` test case as well as properties of each
global mesh in its own module.  Each mesh defines the function
``build_cell_width_lat_lon()``, giving the size of MPAS cells as a function of
longitude and latitude, within its ``__init__.py``.

Each mesh also includes a test case for a series of 1 or more short simulations
that perform dynamic adjustment of the initial condition (currently erroneously
referred to a "spin up"), allowing high amplitude barotropic waves to
dissipate.

.. _dev_ocean_global_ocean_qu240:

QU240
^^^^^

The ``QU240`` mesh is a quasi-uniform mesh with 240-km resolution.  The
``compass.ocean.tests.global_ocean.mesh.qu240`` module includes namelist
options appropriate for forward simulations with both RK4 and split-explicit
time integration on this mesh.  These set the time step and default run
duration for short runs with this mesh.

The default config options for this mesh are:

.. code-block:: cfg

    # Options related to the vertical grid
    [vertical_grid]

    # the type of vertical grid
    grid_type = tanh_dz

    # Number of vertical levels
    vert_levels = 16

    # Depth of the bottom of the ocean
    bottom_depth = 3000.0

    # The minimum layer thickness
    min_layer_thickness = 3.0

    # The maximum layer thickness
    max_layer_thickness = 500.0


    # options for global ocean testcases
    [global_ocean]

    ## config options related to the initial_state step
    # number of cores to use
    init_cores = 4
    # minimum of cores, below which the step fails
    init_min_cores = 1
    # maximum memory usage allowed (in MB)
    init_max_memory = 1000
    # maximum disk usage allowed (in MB)
    init_max_disk = 1000

    ## config options related to the forward steps
    # number of cores to use
    forward_cores = 4
    # minimum of cores, below which the step fails
    forward_min_cores = 1
    # maximum memory usage allowed (in MB)
    forward_max_memory = 1000
    # maximum disk usage allowed (in MB)
    forward_max_disk = 1000

    ## metadata related to the mesh
    # the prefix (e.g. QU, EC, WC, SO)
    prefix = QU
    # a description of the mesh
    mesh_description = MPAS quasi-uniform mesh for E3SM version ${e3sm_version} at
                       ${min_res}-km global resolution with ${levels} vertical
                       level

    # E3SM version that the mesh is intended for
    e3sm_version = 2
    # The revision number of the mesh, which should be incremented each time the
    # mesh is revised
    mesh_revision = 1
    # the minimum (finest) resolution in the mesh
    min_res = 240
    # the maximum (coarsest) resolution in the mesh, can be the same as min_res
    max_res = 240
    # The URL of the pull request documenting the creation of the mesh
    pull_request = <<<Missing>>>

The vertical grid is a ``tanh_dz`` profile (see :ref:`dev_ocean_framework_vertical`)
with 16 vertical levels ranging in thickness from 3 to 500 m.


The ``compass.ocean.tests.global_ocean.mesh.qu240.spinup`` module defines a
test case for performing dynamical adjustment on the mesh.  In the
``damped_spinup_1`` step, the model is run for 1 day with strong Rayleigh
friction (``1e-4`` 1/s) to damp the velocity field.  In the
``simulation`` step, the model runs for an additional 1 day without Rayleigh
friction.  The dynamic adjustment test case takes advantage of Jinja templating
for streams files to use the same streams template for each step in the test
case, see :ref:`dev_streams`.

.. _dev_ocean_global_ocean_quwisc240:

QUwISC240
^^^^^^^^^

This mesh is identical to ``QU240`` in every way except that it includes the
cavities below ice shelves in the ocean domain.  The same dynamical adjustment
test case is used.

.. _dev_ocean_global_ocean_ec30to60:

EC30to60
^^^^^^^^

The ``EC30to60`` mesh is an "eddy-closure" mesh with 30-km resolution at the
equator, 60-km resolution at mid latitudes, and 35-km resolution at the poles.
The mesh resolution is purely a function of latitude and is defined by the
default keyword arguments to
:py:func:`mpas_tools.mesh.creation.mesh_definition_tools.EC_CellWidthVsLat()`
The ``compass.ocean.tests.global_ocean.mesh.ec30to60`` module includes namelist
options appropriate for forward simulations with split-explicit (but not RK4)
time integration on this mesh.  These set the time step and default run
duration for short runs with this mesh.

The default config options for this mesh are:

.. code-block:: cfg

    # Options related to the vertical grid
    [vertical_grid]

    # the type of vertical grid
    grid_type = 60layerPHC


    # options for global ocean testcases
    [global_ocean]

    ## config options related to the initial_state step
    # number of cores to use
    init_cores = 36
    # minimum of cores, below which the step fails
    init_min_cores = 8
    # maximum memory usage allowed (in MB)
    init_max_memory = 1000
    # maximum disk usage allowed (in MB)
    init_max_disk = 1000

    ## config options related to the forward steps
    # number of cores to use
    forward_cores = 128
    # minimum of cores, below which the step fails
    forward_min_cores = 36
    # maximum memory usage allowed (in MB)
    forward_max_memory = 1000
    # maximum disk usage allowed (in MB)
    forward_max_disk = 1000

    ## metadata related to the mesh
    # the prefix (e.g. QU, EC, WC, SO)
    prefix = EC
    # a description of the mesh and initial condition
    mesh_description = MPAS Eddy Closure mesh for E3SM version ${e3sm_version} with
                       enhanced resolution around the equator (30 km), South pole
                       (35 km), Greenland (${min_res} km), ${max_res}-km resolution
                       at mid latitudes, and ${levels} vertical levels
    # E3SM version that the mesh is intended for
    e3sm_version = 2
    # The revision number of the mesh, which should be incremented each time the
    # mesh is revised
    mesh_revision = 3
    # the minimum (finest) resolution in the mesh
    min_res = 30
    # the maximum (coarsest) resolution in the mesh, can be the same as min_res
    max_res = 60
    # The URL of the pull request documenting the creation of the mesh
    pull_request = <<<Missing>>>

The vertical grid is a ``60layerPHC`` profile (see :ref:`dev_ocean_framework_vertical`)
with 60 vertical levels ranging in thickness from 10 to 250 m.

The ``compass.ocean.tests.global_ocean.mesh.ec30to60.spinup`` module defines a
test case for performing dynamical adjustment on the mesh.  In the
``damped_spinup_1`` step, the model is run for 10 days with strong Rayleigh
friction (``1e-4`` 1/s) to damp the velocity field.  In the
``simulation`` step, the model runs for an additional 10 days without Rayleigh
friction.  The dynamic adjustment test case takes advantage of Jinja templating
for streams files to use the same streams template for each step in the test
case, see :ref:`dev_streams`.

.. _dev_ocean_global_ocean_ecwisc30to60:

ECwISC30to60
^^^^^^^^^^^^

This mesh is identical to ``EC30to60`` in every way except that it includes the
cavities below ice shelves in the ocean domain.  The same dynamical adjustment
test case is used.


.. _dev_ocean_global_ocean_sowisc12to60:

SOwISC12to60
^^^^^^^^^^^^

The ``SOwISC12to60`` mesh is a Southern Ocean regionally refined mesh with
12-km resolution around the Southern Ocean and Antarctica, 45-km at southern
mid-latitudes, 30-km at the equator and in the North Atlantic, 60-km resolution
in the North Pacific, and 35-km resolution in the Arctic.
The ``compass.ocean.tests.global_ocean.mesh.so12to60`` module includes namelist
options appropriate for forward simulations with split-explicit (but not RK4)
time integration on this mesh.  These set the time step and default run
duration for short runs with this mesh.

The default config options for this mesh are:

.. code-block:: cfg

    # Options related to the vertical grid
    [vertical_grid]

    # the type of vertical grid
    grid_type = 60layerPHC


    # options for global ocean testcases
    [global_ocean]

    ## config options related to the initial_state step
    # number of cores to use
    init_cores = 36
    # minimum of cores, below which the step fails
    init_min_cores = 8
    # maximum memory usage allowed (in MB)
    init_max_memory = 1000
    # maximum disk usage allowed (in MB)
    init_max_disk = 1000

    ## config options related to the forward steps
    # number of cores to use
    forward_cores = 1296
    # minimum of cores, below which the step fails
    forward_min_cores = 128
    # maximum memory usage allowed (in MB)
    forward_max_memory = 1000
    # maximum disk usage allowed (in MB)
    forward_max_disk = 1000

    ## metadata related to the mesh
    # the prefix (e.g. QU, EC, WC, SO)
    prefix = SO
    # a description of the mesh and initial condition
    mesh_description = MPAS Southern Ocean regionally refined mesh for E3SM version
                       ${e3sm_version} with enhanced resolution (${min_res} km) around
                       Antarctica, 45-km resolution in the mid southern latitudes,
                       30-km resolution in a 15-degree band around the equator, 60-km
                       resolution in northern mid latitudes, 30 km in the north
                       Atlantic and 35 km in the Arctic.  This mesh has ${levels}
                       vertical levels and includes cavities under the ice shelves
                       around Antarctica.
    # E3SM version that the mesh is intended for
    e3sm_version = 2
    # The revision number of the mesh, which should be incremented each time the
    # mesh is revised
    mesh_revision = 4
    # the minimum (finest) resolution in the mesh
    min_res = 12
    # the maximum (coarsest) resolution in the mesh, can be the same as min_res
    max_res = 60
    # The URL of the pull request documenting the creation of the mesh
    pull_request = https://github.com/MPAS-Dev/compass/pull/37


The vertical grid is a ``60layerPHC`` profile (see :ref:`dev_ocean_framework_vertical`)
with 60 vertical levels ranging in thickness from 10 to 250 m.

The ``compass.ocean.tests.global_ocean.mesh.so12to60.spinup`` module defines a
test case for performing dynamical adjustment on the mesh.  In the
``damped_spinup_1`` through ``damped_spinup_3`` steps, the model is run for
2, 4 and 4 days with gradually weakening Rayleigh friction (``1e-4``, ``4e-5``,
and ``1e-5`` 1/s) to damp the velocity field.  In the ``simulation`` step, the
model runs for an additional 10 days without Rayleigh friction.  The
dynamic adjustment test case takes advantage of Jinja templating for streams
files to use the same streams template for each step in the test case, see
:ref:`dev_streams`.

.. _dev_ocean_global_ocean_metadata:

metadata
~~~~~~~~

The module ``compass.ocean.tests.global_ocean.metadata`` determines the values
of a set of metadata related to the E3SM mesh name, initial condition, conda
environment, etc. that are added to nearly all ``global_ocean`` NetCDF output.
See :ref:`global_ocean_metadata` in the User's Guide for more details on
what the metadata looks like.

The values of some of the metadata are given in config options:

.. code-block:: cfg

    # options for global ocean testcases
    [global_ocean]

    ...

    ## metadata related to the mesh
    # whether to add metadata to output files
    add_metadata = True
    # the prefix (e.g. QU, EC, WC, SO)
    prefix = PREFIX
    # a description of the mesh
    mesh_description = <<<Missing>>>
    # a description of the bathymetry
    bathy_description = <<<Missing>>>
    # a description of the mesh with ice-shelf cavities
    init_description = <<<Missing>>>
    # E3SM version that the mesh is intended for
    e3sm_version = 2
    # The revision number of the mesh, which should be incremented each time the
    # mesh is revised
    mesh_revision = <<Missing>>
    # the minimum (finest) resolution in the mesh
    min_res = <<<Missing>>>
    # the maximum (coarsest) resolution in the mesh, can be the same as min_res
    max_res = <<<Missing>>>
    # the maximum depth of the ocean, always detected automatically
    max_depth = autodetect
    # the number of vertical levels, always detected automatically
    levels = autodetect

    # the date the mesh was created as YYMMDD, typically detected automatically
    creation_date = autodetect
    # The following options are detected from .gitconfig if not explicitly entered
    author = autodetect
    email = autodetect
    # The URL of the pull request documenting the creation of the mesh
    pull_request = <<<Missing>>>

Each mesh should define a number of these config options, e.g. ``EC30to60``
defines:

.. code-block:: cfg

    # options for global ocean testcases
    [global_ocean]

    ...

    ## metadata related to the mesh
    # the prefix (e.g. QU, EC, WC, SO)
    prefix = EC
    # a description of the mesh and initial condition
    mesh_description = MPAS Eddy Closure mesh for E3SM version ${e3sm_version} with
                       enhanced resolution around the equator (30 km), South pole
                       (35 km), Greenland (${min_res} km), ${max_res}-km resolution
                       at mid latitudes, and ${levels} vertical levels
    # E3SM version that the mesh is intended for
    e3sm_version = 2
    # The revision number of the mesh, which should be incremented each time the
    # mesh is revised
    mesh_revision = 3
    # the minimum (finest) resolution in the mesh
    min_res = 30
    # the maximum (coarsest) resolution in the mesh, can be the same as min_res
    max_res = 60
    # The URL of the pull request documenting the creation of the mesh
    pull_request = <<<Missing>>>

In this particular case, the ``pull_request`` has not yet been defined.  Each
time the mesh is revised, the ``mesh_revision`` should be updated and the
associated pull request to https://github.com/MPAS-Dev/compass/ should be
added here.

The function :py:func:`compass.ocean.tests.global_ocean.metadata.get_e3sm_mesh_names()`
is used to construct the "short" and "long" names of the mesh using a
standard naming convention for E3SM:

.. code-block:: python

    short_mesh_name = '{}{}E{}r{}'.format(mesh_prefix, res, e3sm_version,
                                          mesh_revision)
    long_mesh_name = '{}{}kmL{}E3SMv{}r{}'.format(mesh_prefix, res, levels,
                                                  e3sm_version, mesh_revision)

For example, the ``QU240`` mesh has the E3SM short name ``QU240E2r1`` and
long name ``QU240kmL16E3SMv2r1``.

.. _dev_ocean_global_ocean_subdir:

subdir
~~~~~~

The subdirectory structure for ``global_ocean`` test cases got a little complex
as we worked to generalize the test cases.  Each test case is in a subdirectory
with the mesh name.  Each test case other than ``mesh`` is in a subdirectory
with information about the initial condition that was used.  Further, test
cases that use the ``RK4`` time integrator are put in their own ``RK4``
subdirectory.

To keep track of all of this without to much effort, the module
``compass.ocean.tests.global_ocean.subdir`` has several helper functions that
test cases can call to find the relative paths to other test cases:

* :py:func:`compass.ocean.tests.global_ocean.subdir.get_init_sudbdir()` gets
  the the subdirectory specific to the initial condition that all test cases
  (other than mesh) are under, for example, ``QU240/PHC``.

* :py:func:`compass.ocean.tests.global_ocean.subdir.get_forward_sudbdir()` gets
  the the subdirectory for a specific test case that includes a forward run.
  If the test case is with the default ``split-explicit`` time integrator, it
  doesn't have a subdirectory to indicate this but if it uses ``RK4``, it is
  placed in an ``RK4`` subdirectory.  Examples:
  ``QU240/EN4_1900/performance_test``, ``QU240/PHC/RK4/restart_test``.

* :py:func:`compass.ocean.tests.global_ocean.subdir.get_mesh_relative_path()`
  gets the relative path to the subdirectory with the mesh name in the
  global_ocean configuration.  For example, for the
  ``QU240/PHC/RK4/restart_test`` test case this would be ``../../..``, but
  for ``QU240/EN4_1900/performance_test``, it would be ``../..``.

* :py:func:`compass.ocean.tests.global_ocean.subdir.get_initial_condition_relative_path()`
  gets the relative path to the subdirectory with the initial condition name in
  the global_ocean configuration.  For example, for the
  ``QU240/PHC/RK4/restart_test`` test case this would be ``../..``, but
  for ``QU240/EN4_1900/performance_test``, it would be ``..``.

The first two functions help with defining subdirectories for test cases to
pass to :py:func:`compass.testcase.set_testcase_subdir()`, while the last two
are helpful for adding outputs from previous test cases as inputs to a given
step, and creating symlinks to these files, e.g.:

.. code-block:: python

    mesh_path = '{}/mesh/mesh'.format(get_mesh_relative_path(step))
    init_path = '{}/init'.format(get_initial_condition_relative_path(step))

    if with_ice_shelf_cavities:
        initial_state_target = '{}/ssh_adjustment/adjusted_init.nc'.format(
            init_path)
    else:
        initial_state_target = '{}/initial_state/initial_state.nc'.format(
            init_path)
    add_input_file(step, filename='init.nc', target=initial_state_target)
    add_input_file(
        step, filename='forcing_data.nc',
        target='{}/initial_state/init_mode_forcing_data.nc'.format(init_path))
    add_input_file(step, filename='graph.info',
                   target='{}/culled_graph.info'.format(mesh_path))

.. _dev_ocean_global_ocean_forward:

forward step
~~~~~~~~~~~~

The module ``compass.ocean.tests.global_ocean.forward`` defines a step
for running MPAS-Ocean from the initial condition produced in the
``initial_state`` step or the output of an ``ssh_adjustment`` step.  When
a test case is adding a forward step with :py:func:`compass.testcase.add_step()`,
it needs to pass a number of keyword arguments to be added to the ``step``
dictionary, e.g.:

.. code-block:: python

    mesh_name = testcase['mesh_name']
    with_ice_shelf_cavities = testcase['with_ice_shelf_cavities']
    initial_condition = testcase['initial_condition']
    with_bgc = testcase['with_bgc']
    time_integrator = testcase['time_integrator']

    ...

    step = add_step(testcase, forward, mesh_name=mesh_name,
                    with_ice_shelf_cavities=with_ice_shelf_cavities,
                    initial_condition=initial_condition, with_bgc=with_bgc,
                    time_integrator=time_integrator)

As in the example above, these are typically passed along from the ``testcase``
dictionary (they were passed in as keyword arguments to
:py:func:`compass.testcase.add_testcase()`).

Performance-related parameters---``cores``, ``min_cores``, ``max_memory``,
``max_disk``, and ``threads``---can be passed as keyword arguments to
``add_step``, but they are more typically read from the corresponding
``forward_<param>`` config options in the ``global_ocean`` section of the
config file.  This lets users update these values as appropriate if the machine
and/or mesh defaults aren't quite right for them.

During ``collect()``, this step adds the ``forward``, ``wisc`` and ``bgc``
namelist replacements and streams files as appropriate based
``with_ice_shelf_cavities`` and ``with_bgc``.

Further namelist replacements and streams files can be added in the test case
after calling ``add_step``, e.g.:

.. code-block:: python

    step = add_step(testcase, forward, mesh_name=mesh_name,
                    with_ice_shelf_cavities=with_ice_shelf_cavities,
                    initial_condition=initial_condition, with_bgc=with_bgc,
                    time_integrator=time_integrator)

    if with_ice_shelf_cavities:
        module = __name__
        add_namelist_file(step, module, 'namelist.wisc')
        add_streams_file(step, module, 'streams.wisc')
        add_output_file(step, filename='land_ice_fluxes.nc')

Namelist and streams files are generate during ``setup()`` and MPAS-Ocean is
run (including updating PIO namelist options and generating a graph partition)
in ``run()``.  Then, :ref:`global_ocean_metadata` is added to the output NetCDF
files.

.. _dev_ocean_global_ocean_testcases:

Test cases
----------

There are 9 ``global_ocean`` test cases.  First, ``mesh`` must be run to
generate and cull the mesh, then one of the variants of ``init`` must be run
to create an initial condition on that mesh.  After that, any of the
regression-focused test cases (``performance_test``, ``restart_test``,
``decomp_test``, ``threads_test``, ``analysis_test``, or ``daily_output_test``)
can be run in any order and as desired.  If an initial condition for E3SM is
desired, the user (or test suite) should first run ``spinup`` and then
``files_for_e3sm``.

.. _dev_ocean_global_ocean_mesh:

mesh test case
~~~~~~~~~~~~~~

This test case generates an MPAS horizontal mesh, then culls out the land cells
to improve model efficiency.

In the ``mesh`` step, first ``build_cell_width_lat_lon()`` is called in the
mesh's module (see :ref:`dev_ocean_global_ocean_meshes`).  Then, the base mesh
is created with :py:func:`mpas_tools.ocean.build_spherical_mesh()`.  Finally,
:py:func:`compass.ocean.tests.global_ocean.mesh.cull.cull_mesh()` is called to
cull the mesh.

The helper function :py:func:`compass.ocean.tests.global_ocean.mesh.mesh.get_mesh_package()`
finds the package for a given mesh based on the mesh name.  This can be useful
for computing the mesh resolution as well as finding the forward namelist
replacements and streams files specific to the mesh, or the ``spinup`` test
case for performing dynamic adjustment of the initial condition.

:py:func:`compass.ocean.tests.global_ocean.mesh.cull.cull_mesh()` uses a number
of capabilities from `MPAS-Tools <http://mpas-dev.github.io/MPAS-Tools/stable/>`_
and `geometric_features <http://mpas-dev.github.io/geometric_features/stable/>`_
to cull the mesh.  It performs the following steps:

1. combining Natural Earth land coverage north of 60S with Antarctic
   ice coverage or grounded ice coverage from BedMachineAntarctica

2. combining transects defining critical passages (if
   ``with_critical_passages=True``)

3. combining points used to seed a flood fill of the global ocean.

4. create masks from land coverage

5. add land-locked cells to land coverage mask.

6. create masks from transects (if ``with_critical_passages=True``)

7. cull cells based on land coverage but with transects present

8. create flood-fill mask based on seeds

9. cull cells based on flood-fill mask

10. create masks from transects on the final culled mesh (if
    ``with_critical_passages=True``)


.. _dev_ocean_global_ocean_init:

init test case
~~~~~~~~~~~~~~

The module ``compass.ocean.tests.global_ocean.init`` defines a test case for
creating a global initial condition using MPAS-Ocean's init mode. Currently
there are two choices for the potential temperature and salinity
fields used for initialization: the Polar science center Hydrographic Climatology
(`PHC <http://psc.apl.washington.edu/nonwp_projects/PHC/Climatology.html>`_)
or the UK MetOffice's EN4 estimated climatology for the year 1900
(`EN4_1900 <https://www.metoffice.gov.uk/hadobs/en4/download-en4-2-0.html>`_).

The test case includes 5 namelist replacement files and 3 streams files.
``namelist.init`` and ``streams.init`` modify the namelist options and set up
the streams needed for the test case, regardless of the particular
configuration.  ``namelist.phc`` and ``namelist.en4_1900`` set namelist options
specific to those two sets of input files.  ``namelist.wisc`` and
``streams.wisc`` configure the test case for meshes that include
:ref:`global_ocean_ice_shelf_cavities`, while ``namelist.bgc`` and
``streams.bgc`` are used to configure the test case when
:ref:`global_ocean_bgc` is included.

The module ``compass.ocean.tests.global_ocean.init.initial_state`` defines
the step for creating the initial state, including defining the topography,
wind stress, shortwave, potential temperature, salinity, and ecosystem input
data files.

The module ``compass.ocean.tests.global_ocean.init.ssh_adjustment`` defines
a step to adjust the ``landIcePressure`` variable to be in closer to dynamical
balance with the sea-surface height (SSH) in configurations with
:ref:`dev_ocean_framework_iceshelf`.

If the test case is being compared with a baseline, the potential temperature,
salinity, and layerThickness are compared with those in the baseline initial
condition to make sure they are identical.  In runs with BGC, a large number
of ecosystem tracers are compared, and in simulations with ice-shelf cavities,
the SSH and land-ice pressure are compared against the baseline.

.. _dev_ocean_global_ocean_performance_test:

performance_test test case
~~~~~~~~~~~~~~~~~~~~~~~~~~

The module ``compass.ocean.tests.global_ocean.performance_test`` defines a test
case for performing a short MPAS-Ocean simulation as a "smoke test" to make
sure nothing is clearly wrong with the configuration.

The module includes ``namelist.wisc`` and ``streams.wisc``, which enable melt
fluxes below ice shelves and write out related fields if the mesh includes
:ref:`dev_ocean_framework_iceshelf`.

If a baseline is provided, prognostic variables as well as ecosystem tracers
(if BGC is active) and ice-shelf melt fluxes (if ice-shelf cavities are
included in the mesh) are compared with a baseline, and the
``time integration`` timer is compared with that of the baseline.

.. _dev_ocean_global_ocean_restart_test:

restart_test test case
~~~~~~~~~~~~~~~~~~~~~~

The module ``compass.ocean.tests.global_ocean.restart_test`` defines a test
case for comparing a ``full_run`` of a longer duration with a ``restart_run``
that is made up of two segments if half the duration with a restart in between.
The length of the full and restart runs depends on the time integrator.  For
the ``split-explicit`` integrator, an 8-hour full run is compared with two
4-hour segments in the restart run.  For the ``RK4`` integrator, the full run
is 20 minutes long, while the restart segments are each 10 minutes.  The test
case ensures that the main prognostic variables---``temperature``,
``salinity``, ``layerThickness`` and ``normalVelocity``---are identical at the
end of the two runs (as well as with a baseline if one is provided when calling
:ref:`dev_compass_setup`).

The various steps and time integrators are configured with
``namelist.<time_integrator>.<step>`` and ``streams.<time_integrator>.<step>``
namelist replacements and streams files.

.. _dev_ocean_global_ocean_decomp_test:

decomp_test test case
~~~~~~~~~~~~~~~~~~~~~

The module ``compass.ocean.tests.global_ocean.decomp_test`` defines a test case
that performs a short run once on 4 cores and once on 8 cores.  It ensures that
``temperature``, ``salinity``, ``layerThickness`` and ``normalVelocity`` are
identical at the end of the two runs (as well as with a baseline if one is
provided when calling :ref:`dev_compass_setup`).

The duration of the run depends on the mesh and time integrator.  For the
:ref:`dev_ocean_global_ocean_qu240` and :ref:`dev_ocean_global_ocean_quwisc240`
(the only meshes that this test case is currently being generated for), the
duration is 6 hours for the ``split-explicit`` integrator and 10 minutes for
``RK4``.

.. _dev_ocean_global_ocean_threads_test:

threads_test test case
~~~~~~~~~~~~~~~~~~~~~~

The module ``compass.ocean.tests.global_ocean.threads_test`` defines a test
case that performs a short run once on 4 cores, each with 1 thread and once
on 4 cores, each with 2 threads.  It ensures that ``temperature``,
``salinity``, ``layerThickness`` and ``normalVelocity`` are identical at the
end of the two runs (as well as with a baseline if one is provided when calling
:ref:`dev_compass_setup`).

The duration of the run depends on the mesh and time integrator.  For the
:ref:`dev_ocean_global_ocean_qu240` and :ref:`dev_ocean_global_ocean_quwisc240`
(the only meshes that this test case is currently being generated for), the
duration is 6 hours for the ``split-explicit`` integrator and 10 minutes for
``RK4``.

.. _dev_ocean_global_ocean_analysis_test:

analysis_test test case
~~~~~~~~~~~~~~~~~~~~~~~

The module ``compass.ocean.tests.global_ocean.analysis_test`` defines a test
case that performs a short run with 14 analysis members (see
:ref:`global_ocean_analysis_test` in the User's Guide). The ``namelist.forward``
and ``streams.forward`` files ensure that the analysis members are enabled and
that the appropriate output is written out.  The test ensures that the
prognostic variables as well as a few variables from each analysis member are
identical to those from the baseline if one is provided when calling
:ref:`dev_compass_setup`.

The duration of the run depends on the mesh and time integrator.  For the
:ref:`dev_ocean_global_ocean_qu240` and :ref:`dev_ocean_global_ocean_quwisc240`
(the only meshes that this test case is currently being generated for), the
duration is 6 hours for the ``split-explicit`` integrator and 10 minutes for
``RK4``.

.. _dev_ocean_global_ocean_daily_output_test:

daily_output_test test case
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The module ``compass.ocean.tests.global_ocean.daily_output_test`` defines a
test case that performs a 1-day run with the ``timeSeriesStatsDaily`` analysis
members (see :ref:`global_ocean_daily_output_test` in the User's Guide). The
``namelist.forward`` and ``streams.forward`` files ensure that the analysis
member are enabled and that the appropriate output (the E3SM defaults for
the ``timeSeriesStatsMonthly`` analysis member) is written out.  The test
ensures that the time average of the prognostic variables as well as the
sea-surface height are identical to those from the baseline if one is provided
when calling :ref:`dev_compass_setup`.

.. _dev_ocean_global_ocean_spinup:

spinup test case
~~~~~~~~~~~~~~~~

The modules ``compass.ocean.tests.global_ocean.mesh.<mesh_name>.spinup`` define
test cases that performs dynamic adjustment of the model in which surface waves
dissipate and the sea-surface height adjusts to be in balance with the dynamic
pressure (see :ref:`global_ocean_spinup` in the User's Guide). Each of the
:ref`global_ocean_meshes` has its own adjustment step, since the needs
(duration of each step, amount of damping, time step, etc.) may be different
between meshes.

Each module includes ``streams.template``, a Jinja2 template for defining
streams (see :ref:`dev_streams`):

.. code-block:: xml

    <streams>

    <stream name="output"
            output_interval="{{ output_interval }}"/>
    <immutable_stream name="restart"
                      filename_template="../restarts/rst.$Y-$M-$D_$h.$m.$s.nc"
                      output_interval="{{ restart_interval }}"/>

    </streams>

Details of the dynamic adjustment steps are discussed under
:ref:`dev_ocean_global_ocean_meshes` above.

.. _dev_ocean_global_ocean_files_for_e3sm:

files_for_e3sm test case
~~~~~~~~~~~~~~~~~~~~~~~~

After running a :ref:`dev_ocean_global_ocean_spinup`, files can be prepared for
use as E3SM ocean and sea-ice initial conditions using the test case defined in
``compass.ocean.tests.global_ocean.files_for_e3sm``.  Output files from the
test case are symlinked in a directory within the test case called
``assembled_files``. See :ref:`global_ocean_files_for_e3sm` in the User's
Guide for more details.  Output file names involve the "mesh short name", see
:ref:`dev_ocean_global_ocean_metadata`.


The test case requires that an argument ``restart_filename`` be passed to
:py:func:`compass.testcase.add_testcase()` (and thus added to the ``testcase``
dictionary).  This should be the final restart file produced by the
:ref:`dev_ocean_global_ocean_spinup` for the given mesh.


The test case is made up of 5 steps:

``compass.ocean.tests.global_ocean.files_for_e3sm.ocean_initial_condition``
    takes out the ``xtime`` variable from the restart file, creating a symlink
    at ``assembled_files/inputdata/ocn/mpas-o/<mesh_short_name>/<mesh_short_name>_no_xtime.nc``

``compass.ocean.tests.global_ocean.files_for_e3sm.ocean_graph_partition``
    computes graph partitions (see :ref:`dev_model`) appropriate for a wide
    range of core counts between ``min_graph_size = int(nCells / 6000)`` and
    ``max_graph_size = int(nCells / 100)``.  Possible processor counts are
    any power of 2 or any multiple of 12, 120 and 1200 in the range.  Symlinks
    to the graph files are placed at
    ``assembled_files/inputdata/ocn/mpas-o/<mesh_short_name>/mpas-o.graph.info.<core_count>``

``compass.ocean.tests.global_ocean.files_for_e3sm.seaice_initial_condition``
    extracts the following variables from the restart file:

    .. code-block:: python

        keep_vars = ['areaCell', 'cellsOnCell', 'edgesOnCell', 'fCell',
                     'indexToCellID', 'latCell', 'lonCell', 'meshDensity',
                     'nEdgesOnCell', 'verticesOnCell', 'xCell', 'yCell', 'zCell',
                     'angleEdge', 'cellsOnEdge', 'dcEdge', 'dvEdge', 'edgesOnEdge',
                     'fEdge', 'indexToEdgeID', 'latEdge', 'lonEdge',
                     'nEdgesOnCell', 'nEdgesOnEdge', 'verticesOnEdge',
                     'weightsOnEdge', 'xEdge', 'yEdge', 'zEdge', 'areaTriangle',
                     'cellsOnVertex', 'edgesOnVertex', 'fVertex',
                     'indexToVertexID', 'kiteAreasOnVertex', 'latVertex',
                     'lonVertex', 'xVertex', 'yVertex', 'zVertex']

        if with_ice_shelf_cavities:
           keep_vars.append('landIceMask')

    A symlink to the resulting file is placed at
    ``assembled_files/inputdata/ocn/mpas-cice/<mesh_short_name>/seaice.<mesh_short_name>_no_xtime.nc``


``compass.ocean.tests.global_ocean.files_for_e3sm.scrip``
    generates a SCRIP file (see :ref:`global_ocean_files_for_e3sm` in the
    User's guide) describing the MPAS-Ocean mesh.  If ice-shelf cavities are
    included, the step also generates a SCRIP file without the ice-shelf
    cavities for use in coupling components that do not interact with ice-shelf
    cavities (atmosphere, land and sea-ice components).

    Symlinks are placed in ``assembled_files/inputdata/ocn/mpas-o/<mesh_short_name>``
    If ice-shelf cavities are present, the two symlinks are named
    ``ocean.<mesh_short_name>.nomask.scrip.<creation_date>.nc``
    and
    ``ocean.<mesh_short_name>.mask.scrip.<creation_date>.nc``.
    Otherwise, only one file is symlinked, and it is named
    ``ocean.<mesh_short_name>.scrip.<creation_date>.nc``

``compass.ocean.tests.global_ocean.files_for_e3sm.diagnostics_files``
    creates mapping files and regions masks for E3SM analysis members and
    `MPAS-Analysis <https://mpas-dev.github.io/MPAS-Analysis/stable/>`_.

    Region masks are created using
    :py:func:`geometric_features.aggregation.get_aggregator_by_name()` for
    the following region groups:

    .. code-block:: python

        region_groups = ['Antarctic Regions', 'Arctic Ocean Regions',
                         'Arctic Sea Ice Regions', 'Ocean Basins',
                         'Ocean Subbasins', 'ISMIP6 Regions',
                         'Transport Transects']

    The resulting region masks are symlinked in the directory
    ``assembled_files/diagnostics/mpas_analysis/region_masks/``
    and named ``<mesh_short_name>_<region_group><ref_date>.nc``

    Masks are also created for the meridional overturning circulation (MOC)
    basins and the transects representing their southern boundaries.
    The resulting region mask is in the same directory as above, and named
    ``<mesh_short_name>_moc_masks_and_transects.nc``

    Mapping files are created from the MPAS-Ocean and -Seaice mesh to 3
    standard comparison grids: a 0.5 x 0.5 degree longitude/latitude grid,
    an Antarctic stereographic grid, and an Arctic stereographic grid.
    The mapping files are symlinked in the directory
    ``assembled_files/diagnostics/mpas_analysis/maps/``
    and named ``map_<mesh_short_name>_to_0.5x0.5degree_bilinear.nc``,
    ``map_<mesh_short_name>_to_6000.0x6000.0km_10.0km_Antarctic_stereo_bilinear.nc``,
    and ``map_<mesh_short_name>_to_6000.0x6000.0km_10.0km_Arctic_stereo_bilinear.nc``.
