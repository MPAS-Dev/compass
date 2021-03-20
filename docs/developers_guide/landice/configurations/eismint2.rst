.. _dev_landice_eismint2:

eismint2
========

The ``eismint2`` configuration implements variants of the EISMINT2 test cases
at 25-km horizontal resolution and with 10 vertical layers (see
:ref:`landice_eismint2`).  Here, we describe the shared framework for this
configuration and the 5 test cases

.. _dev_landice_eismint2_framework:

framework
---------

The shared configuration options for the ``eismint2`` configuration are
described in :ref:`landice_eismint2` in the User's Guide.

Additionally, the configuration has a shared ``namelist.landice`` file with
a several namelist options shared across all test cases, and a shared
``streams.landice`` file that defines ``input``, ``restart``, ``output``, and
``globalStatsOutput`` streams.

setup_mesh
~~~~~~~~~~

The module ``compass.landice.tests.eismint2.setup_mesh`` defines a step for setting
up the mesh for each test case.

The horizontal mesh is constructed at runtime (using
:py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`). A MALI grid is
created with the MPAS-Tools script
``create_landice_grid_from_generic_MPAS_grid.py`` and a graph file is created
to partition the mesh before the model run.

run_experiment
~~~~~~~~~~~~~~

The module ``compass.landice.tests.eismint2.run_experiment`` defines a step for
setting up the initial condition for an EISMINT2 experiment and running MALI
with that initial condition.  For the ``restart_test`` test cases, the model
will run multiple times with different namelist and streams files.  To support
this functionality, this step has an entry in the ``step`` dictionary
``suffixes``, which is a list of suffixes for the these namelist and streams
files.  The model runs once for each suffix.  The default is just ``landice``.

The initial condition is defined, based on the experiment letter, in the
private function ``_setup_eismint2_initial_conditions()``.

.. _dev_landice_eismint2_standard_experiments:

standard_experiments
--------------------

This test sets up the shared mesh, then performs a 200,000-year run for each of
the 6 supported EISMINT2 expeirments (A, B, C, D, F and G) on 4 cores.  It
doesn't contain any :ref:`dev_validation`.

visualize
~~~~~~~~~

The ``compass.landice.tests.eismint2.standard_experiments.visualize`` step
visualizes the results of experiment A by default, but be run manually to plot
the results of any of the 6 experiments.It is control by the config options in
the ``eismint2_viz`` section.

.. _dev_landice_eismint2_decomposition_test:

decomposition_test and enthalpy_decomposition_test
--------------------------------------------------

These test performs a 3,000-year run of experiment F once on 1 core and once on
4 cores.  They ensure that ``thickness``, ``temperature``,
``basalTemperature``, and ``heatDissipation`` are identical at the end of the
two runs (as well as with a baseline if one is provided when calling
:ref:`dev_compass_setup`). The former uses the ``temperature`` thermal solver,
while the latter uses the ``enthalpy`` thermal solver.

.. _dev_landice_eismint2_restart_test:

restart_test and enthalpy_restart_test
--------------------------------------

This test performs a 3,000-year run of experiment F once on 4 cores, then a
sequence of a 2,000-year and then a 1,000-year run on 4 cores.  It ensures that
``thickness``, ``temperature``, ``basalTemperature``, and ``heatDissipation``
are identical at the end of the two runs (as well as with a baseline if one is
provided when calling :ref:`dev_compass_setup`).

The restart step works by creating two different namelist and streams files,
one each with ``landice`` as the suffix and one each with ``landice.rst`` as
the suffix.  The former perform a 2,000-year run from the initial condition,
while the latter perform a 1,000-year restart run beginning with the end of the
first.
