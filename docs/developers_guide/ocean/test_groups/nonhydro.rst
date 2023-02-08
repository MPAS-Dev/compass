.. _dev_ocean_nonhydro:

nonhydro
========

The ``nonhydro`` test group
(:py:class:`compass.ocean.tests.nonhydro.Nonhydro`)
implements 2 test cases (see :ref:`ocean_nonhydro`)
to validate the new nonhydrostatic capability. 
Here, we describe the 2 test cases.

.. _dev_ocean_nonhydro_stratified_seiche:

stratified_seiche
_________________

This test describes an internal stratified seiche. After the creation 
of the mesh and initial conditions, an hydrostatic and a nonhydrostatic 
simulation are run. The ``namelist.forward`` file has a few common namelist 
options for the two models related to run duration, time-stepping scheme, 
horizontal and vertical momentum, and tracer diffusion. 
The two simulations have a shared ``streams.forward`` file that defines 
``mesh``, ``input``, and ``output`` streams. There is also the 
``namelist.nonhydro`` file which defines the PETSc solver, 
preconditioner and tolerances used for the solution of the nonhydrostatic
elliptic problem.

initial_state
~~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.nonhydro.stratified_seiche.initial_state.InitialState`
sets up the initial state for the stratified seiche test case.

First, a planar mesh is generated using :py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`.  
Then, the mesh is culled to remove periodicity in the x direction.  A vertical grid is 
generated, with 100 layers of 0.1m thickness each by default.  Finally, the initial
density profile is computed along with temperature, uniform salinity and zero initial
velocity.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.nonhydro.stratified_seiche.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step.  The ``nonhydro_mode`` argument is a boolean that
determines if the hydrostatic or the nonhydrostatic model is run. 
Namelist and streams files are generate during ``setup()`` and
MPAS-Ocean is run (including updating PIO namelist options and generating a
graph partition) in ``run()``. Both the hydrostatic and nonhydrostatic
simulation are run on 4 cores and have a run duration of 50s, which 
corresponds to one seiche period.

visualize
~~~~~~~~~

The ``visualize`` step defined by
:py:class:`compass.ocean.tests.nonhydro.stratified_seiche.visualize.Visualize`
makes plots of the horizontal and vertical velocity profiles for the hydrostatic
and nonhydrostatic case. The plot shows the horizontal and vertical velocity 
profiles normalized by their respective maxima at locations x=5m and x=0m,
respectively, at time t = 12s.

.. _dev_ocean_nonhydro_solitary_wave:

solitary_wave
_____________

This test describes the evolution of a train of solitary waves. After the creation
of the mesh and initial conditions, an hydrostatic and a nonhydrostatic
simulation are run. The ``namelist.forward`` file has a few common namelist
options for the two models related to run duration, time-stepping scheme, and 
horizontal and vertical tracer diffusion. The files ``namelist.hydro`` and
``namelist.nonhydro`` specify the different momentum viscosities for the two
models, and the latter defines the PETSc solver, preconditioner and tolerances
used for the solution of the nonhydrostatic elliptic problem. The hydrostatic and 
nonhydrostatic run share the same ``streams.forward`` file that defines
``mesh``, ``input``, ``restart``, and ``output`` streams.

initial_state
~~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.nonhydro.solitary_wave.initial_state.InitialState`
sets up the initial state for the solitary test case.

First, a planar mesh is generated using :py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`.
Then, the mesh is culled to remove periodicity in the x direction.  A vertical grid is
generated, with 100 layers of 20m thickness each by default.  Finally, the initial
density profile is computed along with temperature, uniform salinity and zero initial
velocity.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.nonhydro.solitary_wave.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step.  The ``nonhydro_mode`` argument is a boolean that
determines if the hydrostatic or the nonhydrostatic model is run.
Namelist and streams files are generate during ``setup()`` and
MPAS-Ocean is run (including updating PIO namelist options and generating a
graph partition) in ``run()``. Both the hydrostatic and nonhydrostatic
simulation are run on 16 cores and have a run duration of 40h, the time at
which the solitary waves are fully formed.

visualize
~~~~~~~~~

The ``visualize`` step defined by
:py:class:`compass.ocean.tests.nonhydro.solitary_wave.visualize.Visualize`
makes plots of the temperature profile at 40h for the hydrostatic
and nonhydrostatic case. The plot shows that the nonhydrostatic result 
leads to a train of rank-ordered solitary-like internal gravity waves, 
whereas the hydrostatic model fails to capture correct physics. 