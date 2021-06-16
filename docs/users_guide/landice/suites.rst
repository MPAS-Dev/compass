.. _landice_suites:

Test suites
===========

The ``landice`` core currently includes 3 :ref:`test_suites` that can be used
to run a series of land-ice test cases and optionally compare them against a
baseline run of the same tests.

.. _landice_suite_sia_integration:

sia_integration test suite
--------------------------

.. code-block:: bash

    compass suite -c landice -t sia_integration ...

The ``sia_integration`` test suite includes the following test cases:

.. code-block:: none

    landice/dome/2000m/sia_restart_test
    landice/dome/2000m/sia_decomposition_test
    landice/dome/variable_resolution/sia_restart_test
    landice/dome/variable_resolution/sia_decomposition_test
    landice/enthalpy_benchmark/A
    landice/eismint2/decomposition_test
    landice/eismint2/enthalpy_decomposition_test
    landice/eismint2/restart_test
    landice/eismint2/enthalpy_restart_test
    landice/greenland/restart_test
    landice/greenland/decomposition_test
    landice/hydro_radial/restart_test
    landice/hydro_radial/decomposition_test

These are all meant to be short tests that are appropriate for regression
testing with the part of the MALI code that uses the shallow ice approximation
(SIA) and therefore does not require the Albany library.  
These tests should expose problems with the tools for creating
meshes and initial conditions as well as unexpected changes in any operations,
such as advection and diffusion, that affect the 2 main prognostic variables,
ice thickness and velocity, as well as other fields related to ice
thermodynamics.

Several additional tests determine that the model behavior is exactly the same
for a long run or a shorter continuation run from a restart file
(``restart_test``). Others ensure that the code produces the same result when
it is run on different numbers of cores (``decomposition_test``).

.. _landice_suite_fo_integration:

fo_integration test suite
-------------------------

.. code-block:: bash

    compass suite -c landice -t fo_integration ...

The ``fo_integration`` test suite includes the following test cases:

.. code-block:: none

    landice/dome/2000m/fo_decomposition_test
    landice/dome/2000m/fo_restart_test
    landice/dome/variable_resolution/fo_restart_test
    landice/dome/variable_resolution/fo_decomposition_test

All of these tests use the FO velocity solver and require that MALI be compiled
with the Albany library.

.. _landice_suite_full_integration:

full_integration test suite
---------------------------

The ``full_integration`` test suite is a combination of the ``sia_integration``
and ``fo_integration`` test suites.
