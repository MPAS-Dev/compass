.. _dev_ocean_ice_shelf_2d:

ice_shelf_2d
============

The ``ice_shelf_2d`` test group
(:py:class:`compass.ocean.tests.ice_shelf_2d.IceShelf2d`)
implements a very simplified ice-shelf cavity that is invariant in the x
direction (see :ref:`ocean_ice_shelf_2d`). Here, we describe the shared
framework for this test group and the 2 test cases.

framework
---------

The shared config options for the ``ice_shelf_2d`` test group
are described in :ref:`ocean_ice_shelf_2d` in the User's Guide.

Additionally, the test group has a shared ``namelist.forward`` file with
a few common namelist options related to run duration, time step, equation of
state, land-ice fluxes and horizontal viscosity, as well as a shared
``streams.forward`` file that defines ``mesh``, ``input``, ``restart``,
``output`` and ``globalStatsOutput`` streams.

initial_state
~~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.ice_shelf_2d.initial_state.InitialState`
defines a step for setting up the initial state for each test case.

First, a mesh appropriate for the resolution is generated using
:py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`.  Then, the mesh is
culled to remove periodicity in the y direction.  A vertical grid is generated,
with 20 layers of 100-m thickness each by default.  Then, the 1D grid is either
"squashed" down so the sea-surface height corresponds to the location of the
ice-ocean interface (ice draft) using a z-star :ref:`dev_ocean_framework_vertical`
or top layers are removed where there is an ice shelf using a z-level
coordinate. Finally, the initial salinity profile is computed along with
uniform temperature and zero initial velocity.

ssh_adjustment
~~~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.ice_shelf_2d.ssh_adjustment.SshAdjustment`
performs sea-surface height adjustment described
:ref:`dev_ocean_framework_iceshelf`.  Starting from the initial condition
from ``initial_state``, a number of iterations of forward simulation followed
by adjustment of the land-ice pressure field are performed.  The number of
iterations depend on the test case.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.ice_shelf_2d.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step. A link to the MPAS-Ocean executable is created
during ``setup()`` and the MPAS-Ocean is run (including updating PIO namelist
options and generating a graph partition) in ``run()``.

A few namelist options are set and streams are added when frazil is included:

.. code-block:: python

    if with_frazil:
        options = {'config_use_frazil_ice_formation': '.true.',
                   'config_frazil_maximum_depth': '2000.0'}
        self.add_namelist_options(options)
        self.add_streams_file('compass.ocean.streams', 'streams.frazil')

viz
~~~

The class :py:class:`compass.ocean.tests.ice_shelf_2d.viz.Viz` does a
quick-and-dirty visualization of the vertical coordinate (z-star or z-level).

.. _dev_ocean_ice_shelf_2d_default:

default
-------

The :py:class:`compass.ocean.tests.ice_shelf_2d.default.Default` test case
includes the following default config options:

.. code-block:: cfg

    # Options relate to adjusting the sea-surface height or land-ice pressure
    # below ice shelves to they are dynamically consistent with one another
    [ssh_adjustment]

    # the number of iterations of ssh adjustment to perform
    iterations = 15

The test creates and mesh and initial condition, performs 15 iterations of
SSH adjustment to make sure the SSH is as close as possible to being in
dynamic balance with the land-ice pressure.  Then, it performs a 10-minute
(2-time-step) forward simulation. If a baseline is provided when calling
:ref:`dev_compass_setup`, a large number of variables (both prognostic and
related to land-ice fluxes) are checked to make sure they match the baseline.

.. _dev_ocean_ice_shelf_2d_restart_test:

restart_test
------------

The :py:class:`compass.ocean.tests.ice_shelf_2d.restart_test.RestartTest`
includes the following default config options:

.. code-block:: cfg

    # Options relate to adjusting the sea-surface height or land-ice pressure
    # below ice shelves to they are dynamically consistent with one another
    [ssh_adjustment]

    # the number of iterations of ssh adjustment to perform
    iterations = 2

The test creates and mesh and initial condition, performs 2 iterations of
SSH adjustment to make sure the SSH is not too far from dynamic balance with
the land-ice pressure.  (Fewer iterations are performed in this test case
to cut down on total runtime.)

Then, this test performs a 10-minute run once on 4 cores, saving restart files
every time step (every 5 minutes), then it performs a restart run starting at
minute 5 for 5 more minutes.  It ensures that a large number of variables
(prognostic, related to land-ice fluxes, and related to frazil ice formation)
are identical at the end of the two runs (as well as with a baseline if one is
provided when calling :ref:`dev_compass_setup`).

Restart files are saved at the test-case level in the ``restarts`` directory,
rather than within each step, since they will be used across both the ``full``
and ``restart`` steps.

The ``namelist.full`` and ``streams.full`` files are used to set up the run
duration and restart frequency of the full run, while ``namelist.restart`` and
``streams.restart`` make sure that the restart step begins with a restart at
minute 5 and runs for 5 more minutes.
