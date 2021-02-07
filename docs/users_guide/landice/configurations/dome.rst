.. _landice_dome:

dome
====

The ``landice/dome`` configuration implements variants of the dome test case
from `Halfar (1983) <https://doi.org/10.1029/JC088iC10p06043>`_ and
`Bueler et al. (2005) <https://doi.org/10.3189/172756505781829449>`_.

The domain is approximately rectangular with a circularly symmetric ice sheet
at its center.  <<<Further description here>>>

<<<Exmaple image here>>>

Variants of the test case are available at 2000-m uniform or variable
(~1200 m to ~16700 m) horizontal resolution and 10 vertical layers.

The configuration includes 3 test cases.  All test cases have 3 steps,
``setup_mesh``, which defines the mesh and initial conditions for the model;
``run_model`` (given another name in many test cases to distinguish multiple
forward runs), which performs time integration of the model; and ``visualize``,
which optionally plots the results of the test case (PNG files, plot windows,
or both).

config options
--------------

All 3 test cases share the same set of default config options:

.. code-block:: cfg

    # namelist options for dome test cases
    [dome]

    # sizes (in cells) for the 2000m uniform mesh
    nx = 30
    ny = 34

    # resolution (in m) for the 2000m uniform mesh
    dc = 2000.0

    # number of levels in the mesh
    levels = 10

    # the dome type ('halfar' or 'cism')
    dome_type = halfar

    # Whether to center the dome in the center of the cell that is closest to the
    # center of the domain
    put_origin_on_a_cell = True

    # whether to add a small shelf to the test
    shelf = False

    # whether to add hydrology to the initial condition
    hydro = False

    # namelist options related to visualization for dome test cases
    [dome_viz]

    # which time index to visualize
    time_slice = 0

    # whether to save image files
    save_images = True

    # whether to hide figures (typically when save_images = True)
    hide_figs = True

smoke_test
----------

``landice/dome/2000m/smoke_test`` and ``landice/dome/varres/smoke_test`` are
the default version of the dome test case for a short (1 year) test run and
validation of prognostic variables.

decomposition_test
------------------

``landice/dome/2000m/decomposition_test`` and
``landice/dome/varres/decomposition_test`` run short (1 year) integrations
of the model forward in time on 1 (``1proc_run`` step) and then on 4 processors
(``4proc_run`` step) to make sure the resulting prognostic variables are
bit-for-bit identical between the two runs.

restart_test
------------

``landice/dome/2000m/restart_test`` and ``landice/dome/varres/restart_test``
first run a short (2 year) integration of the model forward in time
(``full_run`` step).  Then, a second step (``restart_run``) performs 2
1-year runs, where the second begins from a restart file saved by the first.
Prognostic variables are compared between the "full" and "restart" runs at
year 2 to make sure they are bit-for-bit identical.
