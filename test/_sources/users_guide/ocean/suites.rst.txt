.. _ocean_suites:

Test suites
===========

The ocean core currently includes 5 :ref:`test_suites` that can be used to
run a series of ocean test cases and optionally compare them against a baseline
run of the same tests.

.. _ocean_suite_nightly:

nightly test suite
------------------

.. code-block:: bash

    python -m compass suite -c ocean -t nightly ...

The ``nightly`` test suite includes the following test cases:

.. code-block:: none

    ocean/baroclinic_channel/10km/default
    ocean/baroclinic_channel/10km/threads_test
    ocean/baroclinic_channel/10km/decomp_test
    ocean/baroclinic_channel/10km/restart_test
    ocean/global_ocean/QU240/mesh
    ocean/global_ocean/QU240/PHC/init
    ocean/global_ocean/QU240/PHC/performance_test/split_explicit
    ocean/global_ocean/QU240/PHC/performance_test/RK4
    ocean/global_ocean/QU240/PHC/restart_test/split_explicit
    ocean/global_ocean/QU240/PHC/restart_test/RK4
    ocean/global_ocean/QU240/PHC/decomp_test/split_explicit
    ocean/global_ocean/QU240/PHC/decomp_test/RK4
    ocean/global_ocean/QU240/PHC/threads_test/split_explicit
    ocean/global_ocean/QU240/PHC/threads_test/RK4
    ocean/global_ocean/QU240/PHC/analysis_test/split_explicit
    ocean/global_ocean/QU240/PHC_BGC/init
    ocean/global_ocean/QU240/PHC_BGC/performance_test/split_explicit
    ocean/global_ocean/QU240/PHC_BGC/restart_test/split_explicit
    ocean/global_ocean/QU240/EN4_1900/init
    ocean/global_ocean/QU240/EN4_1900/performance_test/split_explicit
    ocean/ice_shelf_2d/5km/restart_test
    ocean/ziso/20km/default
    ocean/ziso/20km/with_frazil

These are all meant to be short tests that are appropriate for regression
testing.  They cover much (but by no means all) of the features of MPAS-Ocean.
These tests should expose problems with the tools for creating meshes and
initial conditions as well as unexpected changes in any operations, such as
advection and diffusion, that affect the 4 main prognostic variables:
layer thickness, velocity, temperature and salinity.

Several additional tests determine that the model behavior is exactly the same
for a long run or a shorter continuation run from a restart file
(``restart_test``). Others ensure that the code produces the same result when
it is run on different numbers of cores (``decomp_test``) or threads
(``threads_test``). Several tests are run with both the split-explicit and RK4
time integrators (see :ref:`global_ocean_performance_test`).

One test (:ref:`global_ocean_analysis_test`) ensures that the so-called
"analysis members", which can perform analysis as the model runs, are
functioning properly and that their results do not change unexpectedly.

Two tests (``restart_test`` from :ref:`ocean_ice_shelf_2d` and ``with_frazil``
from :ref:`ocean_ziso`) include checks for proper functioning of frazil ice
formation.

One test (``restart_test`` from :ref:`ocean_ice_shelf_2d`) checks that
freshwater fluxes under ice shelves are working as expected.

The ``default`` test from :ref:`ocean_ziso` tests that Lagrangian particles are
working properly.

.. _ocean_suite_quwisc240:

quwisc240 test suite
--------------------

.. code-block:: bash

    python -m compass suite -c ocean -t quwisc240 ...

.. code-block:: none

    ocean/global_ocean/QUwISC240/mesh
    ocean/global_ocean/QUwISC240/PHC/init
    ocean/global_ocean/QUwISC240/PHC/performance_test/split_explicit
    ocean/global_ocean/QUwISC240/PHC/performance_test/RK4
    ocean/global_ocean/QUwISC240/PHC/restart_test/split_explicit
    ocean/global_ocean/QUwISC240/PHC/restart_test/RK4
    ocean/global_ocean/QUwISC240/PHC/decomp_test/split_explicit
    ocean/global_ocean/QUwISC240/PHC/decomp_test/RK4
    ocean/global_ocean/QUwISC240/PHC/threads_test/split_explicit
    ocean/global_ocean/QUwISC240/PHC/threads_test/RK4
    ocean/global_ocean/QUwISC240/PHC/analysis_test/split_explicit
    ocean/global_ocean/QUwISC240/PHC_BGC/init
    ocean/global_ocean/QUwISC240/PHC_BGC/performance_test/split_explicit
    ocean/global_ocean/QUwISC240/PHC_BGC/restart_test/split_explicit
    ocean/global_ocean/QUwISC240/EN4_1900/init
    ocean/global_ocean/QUwISC240/EN4_1900/performance_test/split_explicit

This test suite performs exactly the same set of tests for the QUwISC240 mesh
that are performed on the QU240 mesh in the :ref:`ocean_suite_nightly`.  Since
the QUwISC initial condition is a bit more time consuming to produce and
equilibrate (see :ref:`ocean_ssh_adjustment`), it is not included in the
``nightly`` suite but regression testing on this mesh should also be performed
on a regular basis to ensure no unexpected changes to MPAS-Ocean and E3SM
configurations with ice-shelf cavities.

.. _ocean_suite_qu240_spinups:

qu240_spinups test suite
------------------------

.. code-block:: bash

    python -m compass suite -c ocean -t qu240_spinups ...

.. code-block:: none

    ocean/global_ocean/QU240/mesh
    ocean/global_ocean/QU240/PHC/init
    ocean/global_ocean/QU240/PHC/spinup/split_explicit
    ocean/global_ocean/QU240/PHC/files_for_e3sm/split_explicit

    ocean/global_ocean/QUwISC240/mesh
    ocean/global_ocean/QUwISC240/PHC/init
    ocean/global_ocean/QUwISC240/PHC/spinup/split_explicit
    ocean/global_ocean/QUwISC240/PHC/files_for_e3sm/split_explicit

This suite includes all the tests needed to spin up an initial condition for
E3SM on both the QU240 and QUwISC240 meshes.  They are grouped into a test
suite simply to make them easier to run with one single command.

.. _ocean_suite_ec30to60:

ec30to60 test suite
-------------------

.. code-block:: bash

    python -m compass suite -c ocean -t ec30to60 ...

.. code-block:: none

    ocean/global_ocean/EC30to60/mesh
    ocean/global_ocean/EC30to60/PHC/init
    ocean/global_ocean/EC30to60/PHC/performance_test/split_explicit
    ocean/global_ocean/EC30to60/PHC/spinup/split_explicit
    ocean/global_ocean/EC30to60/PHC/files_for_e3sm/split_explicit
    ocean/global_ocean/EC30to60/EN4_1900/init
    ocean/global_ocean/EC30to60/EN4_1900/performance_test/split_explicit
    ocean/global_ocean/EC30to60/EN4_1900/spinup/split_explicit
    ocean/global_ocean/EC30to60/EN4_1900/files_for_e3sm/split_explicit


This suite is included for convenience so all the tests needed to spin up an
initial condition for E3SM on the EC30to60 mesh can be run with a single
command.

.. _ocean_suite_ecwisc30to60:

ecwisc30to60 test suite
-----------------------

.. code-block:: bash

    python -m compass suite -c ocean -t ec30to60 ...

.. code-block:: none

    ocean/global_ocean/ECwISC30to60/mesh
    ocean/global_ocean/ECwISC30to60/PHC/init
    ocean/global_ocean/ECwISC30to60/PHC/performance_test/split_explicit
    ocean/global_ocean/ECwISC30to60/PHC/spinup/split_explicit
    ocean/global_ocean/ECwISC30to60/PHC/files_for_e3sm/split_explicit
    ocean/global_ocean/ECwISC30to60/EN4_1900/init
    ocean/global_ocean/ECwISC30to60/EN4_1900/performance_test/split_explicit
    ocean/global_ocean/ECwISC30to60/EN4_1900/spinup/split_explicit
    ocean/global_ocean/ECwISC30to60/EN4_1900/files_for_e3sm/split_explicit

Similarly to the previous two suites, this suite is included for convenience so
all the tests needed to spin up an initial condition for E3SM on the
ECwISC30to60 mesh can be run with a single command.
