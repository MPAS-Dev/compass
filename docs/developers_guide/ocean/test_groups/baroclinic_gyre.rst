.. _dev_baroclinic_gyre:

baroclinic_gyre
===========================

The ``baroclinic_gyre`` test group implements variants of the
Baroclinic ocean gyre set-up from the 
`MITgcm test case <https://mitgcm.readthedocs.io/en/latest/examples/baroclinic_gyre/baroclinic_gyre.html>`_.

This test case is described in detail in the User guide (see :ref:`baroclinic_gyre`). Here,
we describe the shared framework for this test group.

Framework
--------------

At this stage, the test case is available at 80-km horizontal
resolution.  By default, the 15 vertical layers vary linearly in thickness with depth, from 50m at the surface to 190m at depth (full depth: 1800m).

The test group includes 2 test cases, called ``performance`` for a short (10-day) run, and ``long`` for a 3-year simulation to run it to quasi-equilibrium.  Both test cases have 2 steps,
``initial_state``, which defines the mesh and initial conditions for the model,
and ``forward``, which performs the time integration of the model.

Additionally, the test group has a shared ``namelist.forward`` file with
a few common namelist options related to horizontal
and vertical momentum and tracer diffusion, as well as a shared
``streams.forward`` file that defines ``mesh``, ``input``, ``restart``, and
``output`` streams. 

initial_state
~~~~~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.baroclinic_gyre.initial_state.InitialState`
defines a step for setting up the initial state for each test case.

First, a mesh appropriate for the resolution is generated using
:py:class:`compass.mesh.QuasiUniformSphericalMeshStep`.  Then, the mesh is
culled to keep a single ocean basin (with lat, lon bounds set in `.cfg` file).  A vertical grid is generated,
with 15 layers of thickness that increases linearly with depth (10m increase by default), from 50m at the surface to 190m at depth (full depth: 1800m).
Finally, the initial temperature field is initialized with a vertical profile to match the discrete values set in the `MITgcm test case <https://mitgcm.readthedocs.io/en/latest/examples/baroclinic_gyre/baroclinic_gyre.html>`_, uniform in horizontal space. Salinity is set to a constant value (34 pus)  and initial
velocity is set to zero. 

The ``initial_state`` step also generates the forcing, defined as zonal wind stress that varies with latitude, surface temperature restoring that varies with latitutde, and writes it to `forcing.nc`.

forward
~~~~~~~~~

The class :py:class:`compass.ocean.tests.baroclinic_gyre.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step.

performance
-------------

``ocean/baroclinic_gyre/80km/performance`` is the default version of the
baroclinic eddies test case for a short (10-day) test run and validation of
prognostic variables for regression testing.  Currently, only the 80-km horizontal
resolution is supported.

long
-----------

``ocean/baroclinic_gyre/80km/long`` performs a longer (3 year) integration
of the model forward in time. The point is to compare the quasi-steady state with theroretical scaling and results from other models. Currently, only the 80-km horizontal
resolution is supported.


