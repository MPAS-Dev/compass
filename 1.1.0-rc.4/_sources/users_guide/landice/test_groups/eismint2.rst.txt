.. _landice_eismint2:

eismint2
========

The ``landice/eismint2`` test group implements variants of the EISMINT2 test
cases from `Payne et al. (2000) <https://doi.org/10.3189/172756500781832891>`_.

The domain is approximately rectangular with a circularly symmetric ice sheet
at its center with a radius of 750 km.  <<<Further description here>>>

<<<Exmaple image here>>>

The test case has a horizontal resolution of 25 km and 10 vertical layers.

The test group includes 5 test cases.  The ``standard_experiments`` test
case implements EISMINT2 experiments A, B, C, D, F and G (but not E).  The
remaining 4 test cases are for regression testing: decomposition and restart
tests with either the `temperature` or the `enthalpy` thermal solver.

config options
--------------

All 3 test cases share the same set of default config options:

.. code-block:: cfg

    # config options for EISMINT2 test cases
    [eismint2]

    # sizes (in cells) for the 25000m uniform mesh
    nx = 64
    ny = 74

    # resolution (in m) for the 25000m uniform mesh
    dc = 25000.0

    # number of levels in the mesh
    levels = 10

    # radius (km) used to cull the mesh
    radius = 750.0


    # config options related to visualization for EISMINT2 test cases
    [eismint2_viz]

    # the name of the experiment or experiments to visualize ('a','b','c','d','f',
    # or 'g'). For multiple experiments, give a comma-separated list
    experiment = a

    # whether to save image files
    save_images = True

    # whether to hide figures (typically when save_images = True)
    hide_figs = True

standard_experiments
--------------------

``landice/eismint2/standard_experiments`` implements 6 of the EISMINT2
experiments: A, B, C, D, F and G (but not E).

==== ============================================ =================
Expt Comment                                      Initial condition
==== ============================================ =================
A    Initial thermomechanical coupling run        Zero ice
B    Stepped 5 K air-temperature warming          Experiment A
C    Stepped change in accumulation rate          Experiment A
D    Stepped change in equilibrium-line altitude  Experiment A
F    Air temperature 15 K cooler                  Zero ice
G    Basal slip throughout                        Zero ice
==== ============================================ =================

Each test case runs for 200,000 years (with 6-month time steps).  A
``visualize`` step can be used to plot the results.  By default, experiment A
is plotted but this can be changed with the ``experiment`` config option in the
``[exismint2_viz]`` sections.

decomposition_test and enthalpy_decomposition_test
--------------------------------------------------

``landice/eismint2/decomposition_test`` and
``landice/eismint2/enthalpy_decomposition_test`` run shorter (3,000-year)
integrations of experiment F forward in time on 1 (``1proc_run`` step) and then
on 4 cores (``4proc_run`` step) to make sure the resulting prognostic variables
are bit-for-bit identical between the two runs.

restart_test and enthalpy_restart_test
--------------------------------------

``landice/eismint2/restart_test`` and ``landice/eismint2/enthalpy_restart_test``
first run a shorter (3,000-year) integration of experiment F forward in time
(``full_run`` step).  Then, a second step (``restart_run``) performs a
2,000-year and then a 1,000-year run, where the second begins from a restart
file saved by the first. Prognostic variables are compared between the "full"
and "restart" runs at year 3,000 to make sure they are bit-for-bit identical.
The restart run is performed with longer (100-year) time steps.
