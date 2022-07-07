.. _dev_landice_enthalpy_benchmark:

enthalpy_benchmark
==================

The ``enthalpy_benchmark`` test group
(:py:class:`compass.landice.tests.enthalpy_benchmark.EnthalpyBenchmark`)
implements variants of the enthalpy benchmark from
`Kleiner et al (2015) <https://doi.org/10.5194/tc-9-217-2015>`_ (see
:ref:`landice_enthalpy_benchmark`).

The test group includes 2 test cases, ``A`` and ``B``, each with 3 or more
steps, ``setup_mesh``, ``run_model`` (possibly run multiple times), and
``visualize``.

.. _dev_landice_enthalpy_benchmark_framework:

framework
---------

The shared config options are described in :ref:`landice_enthalpy_benchmark` in
the User's Guide.

Additionally, the test group has a shared ``namelist.landice`` file with
several common namelist options, such as the time step, run duration, and
thermal solver; and a shared ``streams.landice`` file that defines the
``input``, ``restart``, and ``output`` streams.

setup_mesh
~~~~~~~~~~

The class :py:class:`compass.landice.tests.enthalpy_benchmark.setup_mesh.SetupMesh`
defines a step for setting up the mesh for each test case.

The horizontal mesh is constructed at runtime (using
:py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`).  Then, a MALI grid is
created with the MPAS-Tools script ``create_landice_grid_from_generic_MPAS_grid.py``
and a graph file is created to partition the mesh before the model run.

Finally, the initial condition is defined in the private function
``_setup_dome_initial_conditions()``.

run_model
~~~~~~~~~

The class :py:class:`compass.landice.tests.enthalpy_benchmark.run_model.RunModel`
defines a step for running MALI from the initial condition produced in the
``setup_mesh`` step or from a restart file produced by a previous ``run_model``
step.  In the ``A`` test case, the surface air temperate may be updated before
restart runs as described below.

.. _dev_landice_enthalpy_benchmark_A:

A
-

The :py:class:`compass.landice.tests.enthalpy_benchmark.A.A` test case includes
the following config options :

.. code-block:: cfg

    # namelist options for enthalpy benchmark test cases
    [enthalpy_benchmark]

    # number of levels in the mesh
    levels = 50

    # the initial thickness of the ice sheet (in m)
    thickness = 1000.0

    # the basal heat flux (in W m^{-2})
    basal_heat_flux = 0.042

    # the initial surface air temperature (in K)
    surface_air_temperature = 243.15

    # the initial ice temperature (in K)
    temperature = 243.15

    # the surface air temperature (in K) for the first 100,000 years
    phase1_surface_air_temperature = 243.15

    # the surface air temperature (in K) for the next 50,000 years
    phase2_surface_air_temperature = 268.15

    # the surface air temperature (in K) for the final 150,000 years
    phase3_surface_air_temperature = 243.15

Phase 1 of the test is run for 100,000 years with 25-year time steps and
243.15 K air temperature, phase 2 runs for another 50,000 years with 268.15 K
air temperature, and phase 3 runs for 150,000 more years with the air
temperature back at 243.15 K.

Each of these phases has its own namelist and streams files within the ``A``
test case.

``A`` also contains a ``visualize`` step
(:py:class:`compass.landice.tests.enthalpy_benchmark.A.visualize.Visualize`)
for plotting the results and comparing them to analytic results in a ``.mat``
file.

.. _dev_landice_enthalpy_benchmark_B:

B
-

The :py:class:`compass.landice.tests.enthalpy_benchmark.B.B` test case includes
the following config options :

.. code-block:: cfg

    # namelist options for enthalpy benchmark test cases
    [enthalpy_benchmark]

    # number of levels in the mesh
    levels = 400

    # the initial thickness of the ice sheet (in m)
    thickness = 200.0

    # the basal heat flux (in W m^{-2})
    basal_heat_flux = 0.0

    # the initial surface air temperature (in K)
    surface_air_temperature = 270.15

    # the initial ice temperature (in K)
    temperature = 270.15

To run the ``B`` test case as intended, code modifications are also required
as described in the ``enthalpy_benchmark/README`` file.  This is not a process
that ``compass`` can automate.

This test case runs for 10,000 years with 10-year time steps.

This test case also contains a ``visualize`` step
(:py:class:`compass.landice.tests.enthalpy_benchmark.B.visualize.Visualize`)
for plotting the results and comparing them to analytic results in a ``.mat``
file.
