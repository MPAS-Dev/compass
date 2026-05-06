.. _dev_baroclinic_gyre:

baroclinic_gyre
===============

The ``baroclinic_gyre`` test group implements variants of the
Baroclinic ocean gyre set-up from the 
`MITgcm test case <https://mitgcm.readthedocs.io/en/latest/examples/baroclinic_gyre/baroclinic_gyre.html>`_.

This test case is described in detail in the User guide
(see :ref:`baroclinic_gyre`). Here, we describe the shared framework
for this test group.

Framework
---------

At this stage, the test case is available at 80-km and 20-km horizontal
resolution.  By default, the 15 vertical layers vary linearly in thickness
with depth, from 50m at the surface to 190m at depth (full depth: 1800m).

The test group includes 2 test cases, called ``performance_test`` for a short
(3-time-step) run, and ``3_year_test`` for a 3-year simulation. Note that 
3 years are insufficient to bring the standard test case to full equilibrium.
Both test cases have 2 steps, ``initial_state``, which defines the mesh and 
initial conditions for the model, and ``forward``, which performs the time 
integration of the model.

Additionally, the test group has a shared ``namelist.forward`` file with
a few common namelist options related to horizontal
and vertical momentum and tracer diffusion, as well as a shared
``streams.forward`` file that defines ``mesh``, ``input``, ``restart``, and
``output`` streams. 

cull_mesh
~~~~~~~~~

The class :py:class:`compass.ocean.tests.baroclinic_gyre.cull_mesh.CullMesh`
defines a step for setting up the mesh for the baroclinic gyre test case.

First, a mesh appropriate for the resolution is generated using
:py:class:`compass.mesh.QuasiUniformSphericalMeshStep`.  Then, the mesh is
culled to keep a single ocean basin (with lat, lon bounds set in `.cfg` file).

initial_state
~~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.baroclinic_gyre.initial_state.InitialState`
defines a step for setting up the initial state for each test case.

A vertical grid is generated, with 15 layers of thickness that increases 
linearly with depth (10m increase by default), from 50m at the surface to 190m 
at depth (full depth: 1800m). Finally, the initial temperature field is 
initialized with a vertical profile to approximate the discrete values set in 
the `MITgcm test case <https://mitgcm.readthedocs.io/en/latest/examples/baroclinic_gyre/baroclinic_gyre.html>`_,
uniform in horizontal space. The surface and bottom values are set in the 
`.cfg` file. Salinity is set to a constant value (34 psu, set in the `.cfg` 
file)  and initial velocity is set to zero. 

The ``initial_state`` step also generates the forcing, defined as zonal wind 
stress that varies with latitude, surface temperature restoring that varies 
with latitutde, and writes it to `forcing.nc`.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.baroclinic_gyre.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step.

performance_test
----------------

``ocean/baroclinic_gyre/80km/performance_test`` is the default version of the
baroclinic eddies test case for a short (3 time steps) test run and validation of
prognostic variables for regression testing.  Currently, only the 80-km and 20-km horizontal
resolutions are supported.

3_year_test
-----------

``ocean/baroclinic_gyre/80km/3_year_test`` performs a longer (3 year) integration
of the model forward in time. The point is to (ultimately) compare the quasi-
steady state with theroretical scaling and results from other models. 
Currently, only the 80-km and 20-km horizontal resolutions are supported. 
Note that 3 years is not long enough to reach steady state.
For the 80km configuration, the estimated time to equilibration is roughly 50 years.
For a detailed comparison of the mean state against theory and results from other models,
the mean state at 100 years may be most appropriate to be aligned with the MITgcm results. 

moc
---

The class :py:class:`compass.ocean.tests.baroclinic_gyre.moc.Moc`
defines a step for computing the zonally-averaged meridional overturning
in the baroclinic gyre test case. 

It calculates the moc by summing the monthly mean vertical velocities
in latitude bins (the bin width ``dlat`` is set in the ``.cfg`` file). 
It outputs a netcdf file ``moc.nc`` in the local folder, with the monthly
value of zonally-averaged meriodional overturning in Sv. 

By default, this step is only called after the ``3_year_test`` step since
the ``performance_test`` is too short to output monthly mean output.

viz
---

The class :py:class:`compass.ocean.tests.baroclinic_gyre.viz.Viz`
defines a step for visualizing output from baroclinic gyre.
Currently, it includes 3 separate plots.

First, it plots 2 timeseries to assess the test case spin-up: the top plot is 
layer-mean kinetic energy over 5 different layers (to show dynamical spin-up),
while the bottom plot is layer mean temperature (to show thermal adjustment).
The values plotted are all basin-averaged monthly output. 

It also plots the time-averaged final state of the simulation. The averaging 
period is set up in the ``.cfg`` file (by default, 1 year). Plots include
time-mean sea surface height (SSH), sea surface temperature, and surface heat
flux (in W/m2, due to surface restoring) with contours of SSH superimposed. 
A separate figure is generated showing the time-mean overturning streamfunction,
based on the ``moc.nc`` file generated in the ``moc`` step described above. 

By default, this step is only called after the ``3_year_test`` step since
the ``performance_test`` is too short to output monthly mean output.
