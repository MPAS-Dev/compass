.. _landice_suites:

Test suites
===========

The ``landice`` core currently includes 5 :ref:`test_suites` that can be used
to run a series of land-ice test cases and optionally compare them against a
baseline run of the same tests.  The test suites are described in order of
typical usage.

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

    landice/circular_shelf/decomposition_test
    landice/dome/2000m/fo_decomposition_test
    landice/dome/2000m/fo_restart_test
    landice/dome/variable_resolution/fo_restart_test
    landice/dome/variable_resolution/fo_decomposition_test
    landice/greenland/fo_decomposition_test
    landice/greenland/fo_restart_test
    landice/thwaites/decomposition_test
    landice/thwaites/restart_test
    landice/humboldt/mesh-3km_restart_test/velo-fo_calving-von_mises_stress_damage-threshold_faceMelting

All of these tests use the FO velocity solver and require that MALI be compiled
with the Albany library.

.. _landice_suite_full_integration:

full_integration test suite
---------------------------

The ``full_integration`` test suite is a combination of the ``sia_integration``
and ``fo_integration`` test suites.  This is the suite that should primarily
be used for testing and integration of MALI Pull Requests.

.. _landice_suite_humboldt_calving_tests:

humboldt_calving_tests
----------------------

The ``humboldt_calving_tests`` test suite provide complete coverage of all
calving laws currently support by MALI applied to a 3 km resolution
Humboldt Glacier domain.
For the tests in this suite, the velocity solver is disabled, and the velocity
field comes from an input field, allowing for rapid testing and for testing
bit-for-bit behavior of the calving physics implementations.  
The suite includes:

.. code-block:: none

    landice/humboldt/mesh-3km_decomposition_test/velo-none_calving-none
    landice/humboldt/mesh-3km_decomposition_test/velo-none_calving-floating
    landice/humboldt/mesh-3km_decomposition_test/velo-none_calving-eigencalving
    landice/humboldt/mesh-3km_decomposition_test/velo-none_calving-specified_calving_velocity
    landice/humboldt/mesh-3km_decomposition_test/velo-none_calving-von_mises_stress
    landice/humboldt/mesh-3km_decomposition_test/velo-none_calving-damagecalving
    landice/humboldt/mesh-3km_decomposition_test/velo-none_calving-ismip6_retreat
    landice/humboldt/mesh-3km_decomposition_test/velo-none_calving-von_mises_stress_damage-threshold_faceMelting
    landice/humboldt/mesh-3km_restart_test/velo-none_calving-none
    landice/humboldt/mesh-3km_restart_test/velo-none_calving-floating
    landice/humboldt/mesh-3km_restart_test/velo-none_calving-eigencalving
    landice/humboldt/mesh-3km_restart_test/velo-none_calving-specified_calving_velocity
    landice/humboldt/mesh-3km_restart_test/velo-none_calving-von_mises_stress
    landice/humboldt/mesh-3km_restart_test/velo-none_calving-damagecalving
    landice/humboldt/mesh-3km_restart_test/velo-none_calving-ismip6_retreat
    landice/humboldt/mesh-3km_restart_test/velo-none_calving-von_mises_stress_damage-threshold_faceMelting

.. _landice_suite_humboldt_calving_tests_fo:

humboldt_calving_tests_fo
-------------------------

The ``humboldt_calving_tests_fo`` test suite is identical to
``humboldt_calving_tests`` but with the FO solver enabled.
In this case decomposition tests are not required to be bit-for-bit to pass but
instead use a small tolerance to account for expected differences of the FO
solver on differing numbers of processor.
The suite includes:

.. code-block:: none

    landice/humboldt/mesh-3km_decomposition_test/velo-fo_calving-none
    landice/humboldt/mesh-3km_decomposition_test/velo-fo_calving-floating
    landice/humboldt/mesh-3km_decomposition_test/velo-fo_calving-eigencalving
    landice/humboldt/mesh-3km_decomposition_test/velo-fo_calving-specified_calving_velocity
    landice/humboldt/mesh-3km_decomposition_test/velo-fo_calving-von_mises_stress
    landice/humboldt/mesh-3km_decomposition_test/velo-fo_calving-damagecalving
    landice/humboldt/mesh-3km_decomposition_test/velo-fo_calving-ismip6_retreat
    landice/humboldt/mesh-3km_decomposition_test/velo-fo_calving-von_mises_stress_damage-threshold_faceMelting
    landice/humboldt/mesh-3km_restart_test/velo-fo_calving-none
    landice/humboldt/mesh-3km_restart_test/velo-fo_calving-floating
    landice/humboldt/mesh-3km_restart_test/velo-fo_calving-eigencalving
    landice/humboldt/mesh-3km_restart_test/velo-fo_calving-specified_calving_velocity
    landice/humboldt/mesh-3km_restart_test/velo-fo_calving-von_mises_stress
    landice/humboldt/mesh-3km_restart_test/velo-fo_calving-damagecalving
    landice/humboldt/mesh-3km_restart_test/velo-fo_calving-ismip6_retreat
    landice/humboldt/mesh-3km_restart_test/velo-fo_calving-von_mises_stress_damage-threshold_faceMelting

.. _calving_dt_convergence.txt:

calving_dt_convergence
----------------------

The ``calving_dt_convergence`` test suite runs timestep convergence tests for
calving physics for a number of different model meshes, calving laws, and
velocity solver options.  The tests with "none" velocity solver use data
velocity fiels, and each take 5-10 minutes.  The tests with FO velocity
solver each take a few hours and one may prefer to run them in individual
jobs (which is why they are listed last in the test suite).  Each test
generates a .png image summarizing the results.
The suite includes:

.. code-block:: none

    landice/calving_dt_convergence/mismip+.specified_calving_velocity.none
    landice/calving_dt_convergence/mismip+.von_Mises_stress.none
    landice/calving_dt_convergence/humboldt.specified_calving_velocity.none
    landice/calving_dt_convergence/humboldt.von_Mises_stress.none
    landice/calving_dt_convergence/thwaites.specified_calving_velocity.none
    landice/calving_dt_convergence/thwaites.von_Mises_stress.none
    landice/calving_dt_convergence/mismip+.von_Mises_stress.FO
    landice/calving_dt_convergence/humboldt.von_Mises_stress.FO
    landice/calving_dt_convergence/thwaites.von_Mises_stress.FO
