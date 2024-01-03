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

    compass list --suites

the current set of available test suites is:

.. code-block:: none

    Suites:
      -c landice -t calving_dt_convergence
      -c landice -t fo_integration
      -c landice -t full_integration
      -c landice -t humboldt_calving_tests
      -c landice -t humboldt_calving_tests_fo
      -c landice -t sia_integration
      -c ocean -t cosine_bell_cached_init
      -c ocean -t ec30to60
      -c ocean -t ecwisc30to60
      -c ocean -t kuroshio8to60
      -c ocean -t kuroshio12to60
      -c ocean -t nightly
      -c ocean -t phc
      -c ocean -t pr
      -c ocean -t qu240_for_e3sm
      -c ocean -t quwisc240
      -c ocean -t quwisc240_for_e3sm
      -c ocean -t so12to30
      -c ocean -t sowisc12to30
      -c ocean -t wc14
      -c ocean -t wcwisc14
      -c ocean -t wetdry

As an example, the ocean ``nightly`` test suite includes the test cases used
for regression testing of MPAS-Ocean.  Here are the tests included:

.. code-block:: none

    ocean/baroclinic_channel/10km/default
    ocean/baroclinic_channel/10km/threads_test
    ocean/baroclinic_channel/10km/decomp_test
    ocean/baroclinic_channel/10km/restart_test

    ocean/global_ocean/QU240/mesh
    ocean/global_ocean/QU240/WOA23/init
    ocean/global_ocean/QU240/WOA23/performance_test
    ocean/global_ocean/QU240/WOA23/restart_test
    ocean/global_ocean/QU240/WOA23/decomp_test
    ocean/global_ocean/QU240/WOA23/threads_test
    ocean/global_ocean/QU240/WOA23/analysis_test

    ocean/global_ocean/QU240/WOA23/RK4/performance_test
    ocean/global_ocean/QU240/WOA23/RK4/restart_test
    ocean/global_ocean/QU240/WOA23/RK4/decomp_test
    ocean/global_ocean/QU240/WOA23/RK4/threads_test

    ocean/global_ocean/QUwISC240/mesh
      cached
    ocean/global_ocean/QUwISC240/WOA23/init
      cached
    ocean/global_ocean/QUwISC240/WOA23/performance_test

    ocean/ice_shelf_2d/5km/z-star/restart_test
    ocean/ice_shelf_2d/5km/z-level/restart_test

    ocean/ziso/20km/default
    ocean/ziso/20km/with_frazil

.. note::

    Some tests have "cached" steps, meaning those steps (or the entire test
    case if no specific steps are listed) aren't run but instead the results
    of a previous run are simply downloaded.  This is used to skip steps that
    are prohibitively time consuming during regression testing, but where the
    results are needed to run subsequent tests.  An example above is the
    ``mesh`` and ``WOA23/init`` test cases from the ``ocean/global_ocean/``
    test group on the ``QUwISC240`` mesh.  These tests take several minutes to
    run, which is longer than we wish to take for a quick performance test,
    so they are cached instead.

Including the ``-v`` verbose argument to ``compass list --suites`` will
print the tests belonging to each given suite.
