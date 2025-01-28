.. _dev_landice_eismint2:

eismint2
========

The ``eismint2`` test group (:py:class:`compass.landice.tests.eismint2.Eismint2`)
implements variants of the EISMINT2 test cases at 25-km horizontal resolution
and with 10 vertical layers (see :ref:`landice_eismint2`).  Here, we describe
the shared framework for this test group and the 5 test cases.

.. _dev_landice_eismint2_framework:

framework
---------

The shared config options for the ``eismint2`` test group are
described in :ref:`landice_eismint2` in the User's Guide.

Additionally, the test group has a shared ``namelist.landice`` file with
a several namelist options shared across all test cases, and a shared
``streams.landice`` file that defines ``input``, ``restart``, ``output``, and
``globalStatsOutput`` streams.

setup_mesh
~~~~~~~~~~

The class :py:class:`compass.landice.tests.eismint2.setup_mesh.SetupMesh`
defines a step for setting up the mesh for each test case.

The horizontal mesh is constructed at runtime (using
:py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`). A MALI grid is
created with the MPAS-Tools script
``create_landice_grid_from_generic_MPAS_grid.py`` and a graph file is created
to partition the mesh before the model run.

run_experiment
~~~~~~~~~~~~~~

The class :py:class:`compass.landice.tests.eismint2.run_experiment.RunExperiment`
defines a step for setting up the initial condition for an EISMINT2 experiment
and running MALI with that initial condition.  For the ``restart_test`` test
cases, the model will run multiple times with different namelist and streams
files.  To support this functionality, this step has an attribute ``suffixes``,
which is a list of suffixes for the these namelist and streams files.  The
model runs once for each suffix.  The default is just ``landice``.

The initial condition is defined, based on the experiment letter, in the
private function ``_setup_eismint2_initial_conditions()``.

.. _dev_landice_eismint2_standard_experiments:

standard_experiments
--------------------

The class :py:class:`compass.landice.tests.eismint2.standard_experiments.StandardExperiments`
defines a test that sets up the shared mesh, then performs a 200,000-year run
for each of the 6 supported EISMINT2 experiments (A, B, C, D, F and G) on 4
cores.  It doesn't contain any :ref:`dev_validation`.

visualize
~~~~~~~~~

The :py:class:`compass.landice.tests.eismint2.standard_experiments.visualize.Visualize`
step visualizes the results of experiment A by default, but be run manually to
plot the results of any of the 6 experiments.It is control by the config
options in the ``eismint2_viz`` section.

.. _dev_landice_eismint2_decomposition_test:

decomposition_test and enthalpy_decomposition_test
--------------------------------------------------

The two variants of the
:py:class:`compass.landice.tests.eismint2.decomposition_test.DecompositionTest`
perform a 3,000-year run of experiment F once on 1 core and once on 4 cores.
They ensure that ``thickness``, ``temperature``, ``basalTemperature``, and
``heatDissipation`` are identical at the end of the two runs (as well as with a
baseline if one is provided when calling :ref:`dev_compass_setup`). The default
variant uses the ``temperature`` thermal solver, while the ``enthalpy``
variant uses the ``enthalpy`` thermal solver.

.. _dev_landice_eismint2_restart_test:

restart_test and enthalpy_restart_test
--------------------------------------

The two variants of the
:py:class:`compass.landice.tests.eismint2.restart_test.RestartTest`
perform a 3,000-year run of experiment F once on 4 cores, then a sequence of a
2,000-year and then a 1,000-year run on 4 cores.  They ensures that
``thickness``, ``temperature``, ``basalTemperature``, and ``heatDissipation``
are identical at the end of the two runs (as well as with a baseline if one is
provided when calling :ref:`dev_compass_setup`). The default variant uses the
``temperature`` thermal solver, while the ``enthalpy`` variant uses the
``enthalpy`` thermal solver.

The restart step works by creating two different namelist and streams files,
one each with ``landice`` as the suffix and one each with ``landice.rst`` as
the suffix.  The former perform a 2,000-year run from the initial condition,
while the latter perform a 1,000-year restart run beginning with the end of the
first.
