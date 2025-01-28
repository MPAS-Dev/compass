.. _dev_landice_hydro_radial:

hydro_radial
============

The ``hydro_radial`` test group (:py:class:`compass.landice.tests.hydro_radial.HydroRadial`)
implements variants of the radially symmetric hydrological test case at 1-km
uniform resolution (see :ref:`landice_hydro_radial`).
Here, we describe the shared framework for this test group and the 4 test
cases.

.. _dev_landice_hydro_radial_framework:

framework
---------

The shared config options for the ``hydro_radial`` test group are described in
:ref:`landice_hydro_radial` in the User's Guide.

Additionally, the test group has a shared ``namelist.landice`` file with
a several shared namelist options related to time step, run duration, and
hydrology model, as well as a shared ``streams.landice`` file that defines
``input``, ``restart``, and ``output`` streams.

setup_mesh
~~~~~~~~~~

The class :py:class:`compass.landice.tests.hydro_radial.setup_mesh.SetupMesh`
defines a step for setting up the mesh for each test case.

The horizontal mesh is constructed at runtime (using
:py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`). A MALI grid is
created with the MPAS-Tools script
``create_landice_grid_from_generic_MPAS_grid.py`` and a graph file is created
to partition the mesh before the model run.

Finally, the initial condition is defined in the private function
``_setup_hydro_radial_initial_conditions()``.  Most test cases use a
``zero`` initial condition (meaning a very thin initial water thickness),
but the
``steady_state_drift_test`` uses an ``exact`` initial condition that uses
a nearly exact numerical solution in steady state, as described in
`Bueler et al. (2015) <https://doi.org/10.5194/gmd-8-1613-2015>`_.

run_model
~~~~~~~~~

The class :py:class:`compass.landice.tests.hydro_radial.run_model.RunModel`
defines a step for running MALI from the initial condition produced in the
``setup_mesh`` step. For the ``restart_test`` test cases, the model will run
multiple times with different namelist and streams files.  To support this
functionality, this step has an attribute ``suffixes``, which is a list of
suffixes for the these namelist and streams files.  The model runs once for
each suffix.  The default is just ``landice``.

visualize
~~~~~~~~~

The :py:class:`compass.landice.tests.hydro_radial.visualize.Visualize` step
is optional in each test case and can be run manually to plot the results of
the test case.  It iscontrol by the config options in the ``hydro_radial_viz``
section.

.. _dev_landice_hydro_radial_spinup_test:

spinup_test
-----------

The :py:class:`compass.landice.tests.hydro_radial.spinup_test.SpinupTest`
performs a 10,000-year run from ``zero`` initial conditions on 4 cores.  It
doesn't contain any :ref:`dev_validation`.

.. _dev_landice_hydro_radial_steady_state_drift_test:

steady_state_drift_test
-----------------------

The :py:class:`compass.landice.tests.hydro_radial.steady_state_drift_test.SteadyStateDriftTest`
performs a 1-month run from ``exact`` initial conditions on 4 cores. It doesn't
contain any :ref:`dev_validation`.


.. _dev_landice_hydro_radial_decomposition_test:

decomposition_test
------------------

The :py:class:`compass.landice.tests.hydro_radial.decomposition_test.DecompositionTest`
performs a 1-month run once on 1 core and once on 3 cores.  It ensures that
``waterThickness`` and ``waterPressure`` are identical at the end of the two
runs (as well as with a baseline if one is provided when calling
:ref:`dev_compass_setup`).

.. _dev_landice_hydro_radial_restart_test:

restart_test
------------

The :py:class:`compass.landice.tests.hydro_radial.restart_test.RestartTest`
performs a 2-month run once, then a sequence of 2 1-month runs.
It ensures that ``waterThickness`` and ``waterPressure`` are identical
at the end of the two runs (as well as with a baseline if one is provided when
calling :ref:`dev_compass_setup`).

The restart step works by creating two different namelist and streams files,
one each with ``landice`` as the suffix and one each with ``landice.rst`` as
the suffix.  The former perform a 1-year run from the initial condition, while
the latter perform a 1-year restart run beginning with the end of the first.
