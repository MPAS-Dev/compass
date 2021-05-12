.. _test_suites:

Test Suites
===========

In ``compass``, test suites are simply lists of test cases to be run together
in one operation.  One common reason for running a test suite is to check for
changes in performance or output data compared with a previous run of the
same suite.  This type of
`regression testing <https://en.wikipedia.org/wiki/Regression_testing>`_ is one
of the primary reasons that compass exists. Another reason to define a test
suite is simply to make it easier to run a sequence of test cases (e.g. from
the same test group) that are often run together.

Test suites are defined by their MPAS core and name.  As you can see by
running:

.. code-block:: bash

    python -m compass list --suites

the current set of available test suites is:

.. code-block:: none

    Suites:
      -c landice -t sia_integration
      -c ocean -t nightly
      -c ocean -t ecwisc30to60
      -c ocean -t quwisc240_for_e3sm
      -c ocean -t quwisc240
      -c ocean -t ec30to60
      -c ocean -t qu240_for_e3sm

As an example, the ocean ``nightly`` test suite includes the test cases used
for regression testing of MPAS-Ocean.  Here are the tests included:

.. code-block:: none

    ocean/baroclinic_channel/10km/default
    ocean/baroclinic_channel/10km/threads_test
    ocean/baroclinic_channel/10km/decomp_test
    ocean/baroclinic_channel/10km/restart_test
    ocean/global_ocean/QU240/mesh
    ocean/global_ocean/QU240/PHC/init
    ocean/global_ocean/QU240/PHC/performance_test
    ocean/global_ocean/QU240/PHC/restart_test
    ocean/global_ocean/QU240/PHC/decomp_test
    ocean/global_ocean/QU240/PHC/threads_test
    ocean/global_ocean/QU240/PHC/analysis_test
    ocean/global_ocean/QU240/PHC/RK4/performance_test
    ocean/global_ocean/QU240/PHC/RK4/restart_test
    ocean/global_ocean/QU240/PHC/RK4/decomp_test
    ocean/global_ocean/QU240/PHC/RK4/threads_test
    ocean/global_ocean/QU240/EN4_1900/init
    ocean/global_ocean/QU240/EN4_1900/performance_test
    ocean/global_ocean/QU240/PHC_BGC/init
    ocean/global_ocean/QU240/PHC_BGC/performance_test
    ocean/ice_shelf_2d/5km/restart_test
    ocean/ziso/20km/default
    ocean/ziso/20km/with_frazil

After setting up a test case, you can see the tests in the case you opening
the file ``<suite>.txt`` in the work directory.  You can remove or comment out
(with ``#``) any tests you don't want to run before running the suite.

To run the suite, call:

.. code-block:: bash

    compass run

If you have set up multiple suites in the same directory, run:

.. code-block:: bash

    compass run <suite>

to select a specific suite to run.
