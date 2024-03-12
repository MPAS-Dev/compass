.. _ocean_suites:

Test suites
===========

The ocean core currently includes 16 :ref:`test_suites` that can be used to
run a series of ocean test cases and optionally compare them against a baseline
run of the same tests.  Several are described below but several are missing
from the documentation and should be added.

.. _ocean_suite_nightly:

nightly test suite
------------------

.. code-block:: bash

    compass suite -s -c ocean -t nightly ...

The ``nightly`` test suite includes the following test cases:

.. code-block:: none

    ocean/baroclinic_channel/10km/default
    ocean/baroclinic_channel/10km/threads_test
    ocean/baroclinic_channel/10km/decomp_test
    ocean/baroclinic_channel/10km/restart_test

    ocean/global_ocean/Icos240/mesh
    ocean/global_ocean/Icos240/WOA23/init
    ocean/global_ocean/Icos240/WOA23/performance_test
    ocean/global_ocean/Icos240/WOA23/restart_test
    ocean/global_ocean/Icos240/WOA23/decomp_test
    ocean/global_ocean/Icos240/WOA23/threads_test
    ocean/global_ocean/Icos240/WOA23/analysis_test

    ocean/global_ocean/Icos240/WOA23/RK4/performance_test
    ocean/global_ocean/Icos240/WOA23/RK4/restart_test
    ocean/global_ocean/Icos240/WOA23/RK4/decomp_test
    ocean/global_ocean/Icos240/WOA23/RK4/threads_test

    ocean/global_ocean/IcoswISC240/mesh
      cached
    ocean/global_ocean/IcoswISC240/WOA23/init
      cached
    ocean/global_ocean/IcoswISC240/WOA23/performance_test

    ocean/ice_shelf_2d/5km/z-star/restart_test
    ocean/ice_shelf_2d/5km/z-level/restart_test

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

    compass suite -s -c ocean -t quwisc240 ...

.. code-block:: none

    ocean/global_ocean/QUwISC240/mesh
    ocean/global_ocean/QUwISC240/WOA23/init
    ocean/global_ocean/QUwISC240/WOA23/performance_test
    ocean/global_ocean/QUwISC240/WOA23/restart_test
    ocean/global_ocean/QUwISC240/WOA23/decomp_test
    ocean/global_ocean/QUwISC240/WOA23/threads_test
    ocean/global_ocean/QUwISC240/WOA23/analysis_test
    ocean/global_ocean/QUwISC240/WOA23/RK4/performance_test
    ocean/global_ocean/QUwISC240/WOA23/RK4/restart_test
    ocean/global_ocean/QUwISC240/WOA23/RK4/decomp_test
    ocean/global_ocean/QUwISC240/WOA23/RK4/threads_test
    ocean/global_ocean/QUwISC240/EN4_1900/init
    ocean/global_ocean/QUwISC240/EN4_1900/performance_test

This test suite performs exactly the same set of tests for the QUwISC240 mesh
that are performed on the QU240 mesh in the :ref:`ocean_suite_nightly`.  Since
the QUwISC240 initial condition is a bit more time consuming to produce and
equilibrate (see :ref:`ocean_ssh_adjustment`), it is not included in the
``nightly`` suite but regression testing on this mesh should also be performed
on a regular basis to ensure no unexpected changes to MPAS-Ocean and E3SM
configurations with ice-shelf cavities.

.. _ocean_suite_pr:

pr test suite
-------------

.. code-block:: bash

    compass suite -s -c ocean -t pr ...

The ``pr`` test suite includes the following test cases:

.. code-block:: none


    ocean/baroclinic_channel/10km/default
    ocean/baroclinic_channel/10km/threads_test
    ocean/baroclinic_channel/10km/decomp_test
    ocean/baroclinic_channel/10km/restart_test

    ocean/internal_wave/default
    ocean/internal_wave/vlr/default

    ocean/global_convergence/qu/cosine_bell
      cached: QU60_mesh QU60_init QU90_mesh QU90_init QU120_mesh QU120_init
      cached: QU150_mesh QU150_init QU180_mesh QU180_init QU210_mesh QU210_init
      cached: QU240_mesh QU240_init

    ocean/global_ocean/Icos240/mesh
    ocean/global_ocean/Icos240/WOA23/init
    ocean/global_ocean/Icos240/WOA23/performance_test
    ocean/global_ocean/Icos240/WOA23/restart_test
    ocean/global_ocean/Icos240/WOA23/decomp_test
    ocean/global_ocean/Icos240/WOA23/threads_test
    ocean/global_ocean/Icos240/WOA23/analysis_test
    ocean/global_ocean/Icos240/WOA23/dynamic_adjustment

    ocean/global_ocean/Icos240/WOA23/RK4/performance_test
    ocean/global_ocean/Icos240/WOA23/RK4/restart_test
    ocean/global_ocean/Icos240/WOA23/RK4/decomp_test
    ocean/global_ocean/Icos240/WOA23/RK4/threads_test

    ocean/global_ocean/IcoswISC240/mesh
      cached
    ocean/global_ocean/IcoswISC240/WOA23/init
      cached
    ocean/global_ocean/IcoswISC240/WOA23/performance_test

    ocean/global_ocean/Icos/mesh
      cached
    ocean/global_ocean/Icos/WOA23/init
      cached
    ocean/global_ocean/Icos/WOA23/performance_test

    ocean/global_ocean/IcoswISC/mesh
      cached
    ocean/global_ocean/IcoswISC/WOA23/init
      cached
    ocean/global_ocean/IcoswISC/WOA23/performance_test

    ocean/ice_shelf_2d/5km/z-star/restart_test
    ocean/ice_shelf_2d/5km/z-level/restart_test

    ocean/isomip_plus/planar/2km/z-star/Ocean0

    ocean/ziso/20km/default
    ocean/ziso/20km/with_frazil

These are all meant to be slightly more comprehensive tests than ``nightly``,
to be compared to a baseline before a compass or MPAS-Ocean PR gets merged.
They cover additional features such as convergence, higher resolution meshes,
and vertical Lagrangian remapping.


.. _ocean_suite_qu240_for_e3sm:

qu240_for_e3sm test suite
-------------------------

.. code-block:: bash

    compass suite -c ocean -t qu240_for_e3sm ...

.. code-block:: none

    ocean/global_ocean/QU240/mesh
    ocean/global_ocean/QU240/WOA23/init
    ocean/global_ocean/QU240/WOA23/dynamic_adjustment
    ocean/global_ocean/QU240/WOA23/files_for_e3sm

This suite includes all the tests needed to spin up an initial condition for
E3SM on the QU240 mesh.

.. _ocean_suite_quwisc240_for_e3sm:

quwisc240_for_e3sm test suite
-----------------------------

.. code-block:: bash

    compass suite -c ocean -t quwisc240_for_e3sm ...

.. code-block:: none

    ocean/global_ocean/QUwISC240/mesh
    ocean/global_ocean/QUwISC240/WOA23/init
    ocean/global_ocean/QUwISC240/WOA23/dynamic_adjustment
    ocean/global_ocean/QUwISC240/WOA23/files_for_e3sm

This suite includes all the tests needed to spin up an initial condition for
E3SM on the QUwISC240 mesh.

.. _ocean_suite_ec30to60:

ec30to60 test suite
-------------------

.. code-block:: bash

    compass suite -c ocean -t ec30to60 ...

.. code-block:: none

    ocean/global_ocean/EC30to60/mesh
    ocean/global_ocean/EC30to60/WOA23/init
    ocean/global_ocean/EC30to60/WOA23/performance_test
    ocean/global_ocean/EC30to60/WOA23/dynamic_adjustment
    ocean/global_ocean/EC30to60/WOA23/files_for_e3sm

This suite is included for convenience so all the tests needed to spin up an
initial condition for E3SM on the EC30to60 mesh can be run with a single
command.  A short performance test is also included.

.. _ocean_suite_ecwisc30to60:

ecwisc30to60 test suite
-----------------------

.. code-block:: bash

    compass suite -c ocean -t ec30to60 ...

.. code-block:: none

    ocean/global_ocean/ECwISC30to60/mesh
    ocean/global_ocean/ECwISC30to60/WOA23/init
    ocean/global_ocean/ECwISC30to60/WOA23/performance_test
    ocean/global_ocean/ECwISC30to60/WOA23/dynamic_adjustment
    ocean/global_ocean/ECwISC30to60/WOA23/files_for_e3sm

Similarly to the previous 3 suites, this suite is included for convenience so
all the tests needed to spin up an initial condition for E3SM on the
ECwISC30to60 mesh can be run with a single command.   A short performance test
is also included.
