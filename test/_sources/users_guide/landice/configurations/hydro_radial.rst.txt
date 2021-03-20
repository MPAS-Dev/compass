.. _landice_hydro_radial:

hydro_radial
============

The ``landice/hydro_radial`` configuration implements variants of the
radially symmetric hydrological test case from
`Bueler et al. (2015) <https://doi.org/10.5194/gmd-8-1613-2015>`_.

The domain is approximately rectangular with a radially symmetric ice sheet
at its center.  The mesh has 1-km horizontal resolution and 3 vertical layers.
<<<Further description here>>>

<<<Exmaple image here>>>

The configuration includes 4 test cases.  All test cases are made up of 3
types of steps, ``setup_mesh``, which defines the mesh and initial conditions
for the model; ``run_model`` (given another name in many test cases to
distinguish multiple forward runs), which performs time integration of the
model; and ``visualize``, which optionally plots the results of the test case
(PNG files, plot windows, or both).

config options
--------------

All 4 test cases share the same set of default config options:

.. code-block:: cfg

    # config options for hydro_radial test cases
    [hydro_radial]

    # sizes (in cells) for the 1000m uniform mesh
    nx = 50
    ny = 58

    # resolution (in m) for the 1000m uniform mesh
    dc = 1000.0

    # number of levels in the mesh
    levels = 3


    # config options related to visualization for hydro_radial test cases
    [hydro_radial_viz]

    # which time index to visualize
    time_slice = -1

    # whether to save image files
    save_images = True

    # whether to hide figures (typically when save_images = True)
    hide_figs = True


spinup_test
-----------

``landice/hydro_radial/spinup_test`` performs a 10,000-year spin-up run from
"zero" (a very thin ice sheet) to quasi-steady state.

steady_state_drift_test
-----------------------

``landice/hydro_radial/steady_state_drift_test`` performs a 1-month run,
starting from the "exact" (that is, a nearly exact steady-state solution) as
the initial condition.

decomposition_test
------------------

``landice/hydro_radial/decomposition_test`` runs short (1-month) integrations
of the model forward in time on 1 (``1proc_run`` step) and then on 3 cores
(``3proc_run`` step) to make sure the resulting prognostic variables are
bit-for-bit identical between the two runs.

restart_test
------------

``landice/hydro_radial/restart_test`` first run a 2-year integration of the
model forward in time (``full_run`` step).  Then, a second step
(``restart_run``) performs 2 1-year runs, where the second begins from a
restart file saved by the first. Prognostic variables are compared between the
"full" and "restart" runs at year 2 to make sure they are bit-for-bit
identical.
