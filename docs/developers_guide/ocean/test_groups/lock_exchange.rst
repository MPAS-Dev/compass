.. _dev_ocean_lock_exchange:

lock_exchange
=============

The ``lock_exchange`` test group
(:py:class:`compass.ocean.tests.lock_exchange.LockExchange`)
implements two test cases (see :ref:`ocean_lock_exchange`)
which represent a lock exchange scenario, i.e. two fluids
with different densities that interact over time. 
The first test uses the standard hydrostatic ocean model,
whereas the second one uses the nonhydrostatic version.
Here, we describe the shared framework for this test group 
and the two test cases.

.. _dev_ocean_lock_exchange_framework:

framework
---------

The shared config options for the ``lock_exchange`` test group
are described in :ref:`ocean_lock_exchange` in the User's Guide.

Additionally, the test group has a shared ``namelist.forward`` file with
a few common namelist options related to time step, run duration, viscosity,
and drag, as well as a shared ``streams.forward`` file that defines ``mesh``,
``input``, and ``output`` streams.

initial_state
~~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.lock_exchange.initial_state.InitialState`
sets up the initial state for the two test case.

First, a planar mesh is generated using :py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`.
Then, the mesh is culled to remove periodicity in the x direction.  A vertical grid is
generated, with 100 layers of 0.001m thickness each by default.  Finally, the initial
density profile is computed along with temperature, uniform salinity and zero initial
velocity.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.lock_exchange.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step. Namelist and streams files are generate during 
``__init__()`` and MPAS-Ocean is run (including updating PIO namelist options 
and generating a graph partition) in ``run()``. The simulation (either
hydrostatic or nonhydrostatic) is run on 16 cores using the RK4 time-stepping
scheme and has a run duration of 5s. The ``namelist.forward`` file has a few 
common namelist options for the two test cases related to run duration, 
time-stepping scheme, horizontal and vertical momentum, and tracer diffusion.
The two cases have a shared ``streams.forward`` file that defines
``mesh``, ``input``, and ``output`` streams.

visualize
~~~~~~~~~

The ``visualize`` step defined by
:py:class:`compass.ocean.tests.lock_exchange.visualize.Visualize`
makes a plot of the densitiy profile at time t = 5s. 

.. _dev_ocean_lock_exchange_hydro:

hydro
_____

This test uses the standard hydrostatic model to describe the interaction 
over time of two basins of water with different densities. After the creation 
of the mesh and initial conditions, a hydrostatic simulation is run. 
The density plot shows that the density fronts cannot develop in the upper 
and lower layer. This happens because of the hydrostatic assumption, which 
prevents the generation of the Kelvin-Helmholtz instability.

.. _dev_ocean_lock_exchange_nonhydro:

nonhydro
________

This test uses the nonhydrostatic model to describe the interaction
over time of two basins of water with different densities. After the creation
of the mesh and initial conditions, a nonhydrostatic simulation is run.
The file ``namelist.nonhydro`` defines the PETSc solver, preconditioner 
and tolerances used for the solution of the nonhydrostatic elliptic problem.
The density plot shows that a Kelvin-Helmholtz instability generates with the
nonhydrostatic model, which causes vigorous turbulent mixing to develop on the
interface between high and low-density water.
