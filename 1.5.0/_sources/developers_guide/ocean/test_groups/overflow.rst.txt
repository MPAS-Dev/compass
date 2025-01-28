.. _dev_ocean_overflow:

overflow
========

The ``overflow`` test group
(:py:class:`compass.ocean.tests.overflow.Overflow`)
implements variants of the continental shelf overflow test case.
It is composed of four test cases. The first two, ``default``
and ``rpe_test``, have a shared framework and use the standard
hydrostatic model to investigate the impact on the solution of
different values of the viscosity. The remaining two tests,
``nonhydro`` and ``hydro_vs_nonhydro``, use a different mesh
and initial conditions to explore the impact of a 
nonhydrostatic formulation in an overflow scenario.
Here, we describe the four test cases starting with the 
shared framework for the first two.

.. _dev_ocean_overflow_framework:

framework
---------

The shared config options for the ``overflow`` test group
are described in :ref:`ocean_overflow` in the User's Guide.

Additionally, the test group has a shared ``namelist.forward`` file with
a few common namelist options related to time step, run duration, viscosity,
and drag, as well as a shared ``streams.forward`` file that defines ``mesh``,
``input``, and ``output`` streams.

initial_state_from_init_mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.overflow.initial_state_from_init_mode.InitialStateFromInitMode`
defines a step for setting up the initial state for each test case.

First, a mesh appropriate for the resolution is generated using
:py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`.  Then, the mesh is
culled to remove periodicity in the y direction. The ocean model is then run
in ``init`` mode to generate the vertical grid and populate initial conditions.

initial_state
~~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.overflow.initial_state.InitialState`
sets up the initial state for the ``nonhydro`` and ``hydro_vs_nonhydro`` test cases.

First, a planar mesh is generated using :py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`.
Then, the mesh is culled to remove periodicity in the x direction. The domain is
200m deep and 6.4 km across. After the topography is specified, a vertical grid is generated
with 60 layers.  Finally, the initial density and temperature profiles are computed along with
uniform salinity and zero initial velocity.
 
forward
~~~~~~~

The class :py:class:`compass.ocean.tests.overflow.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state_from_init_mode`` step. If ``nu`` is provided as an argument 
to the constructor, the associate namelist option (``config_mom_del2``) will be
given this value. Namelist and streams files are also generated. MPAS-Ocean is 
run (including updating PIO namelist options and generating a graph partition) 
in ``run()``.

.. _dev_ocean_overflow_default:

default
-------

The :py:class:`compass.ocean.tests.overflow.default.Default`
test performs a 12-minute run on 4 cores.  It doesn't contain any
:ref:`dev_validation`.

.. _dev_ocean_overflow_rpe_test:

rpe_test
--------

The :py:class:`compass.ocean.tests.overflow.rpe_test.RpeTest`
performs a longer (40 hour) integration of the model forward in time at 5
different values of the viscosity.  These ``nu`` values are later added as
arguments to the ``Forward`` steps' constructors when they are added to the
test case:

.. code-block:: python

            step = Forward(
                test_case=self, name=name, subdir=name, ntasks=4,
                openmp_threads=1, nu=float(nu))
            ...
            self.add_step(step)

The ``analysis`` step defined by
:py:class:`compass.ocean.tests.overflow.rpe_test.analysis.Analysis`
makes plots of the final results with each value of the viscosity.

This test is resource intensive enough that it is not used in regression
testing.

nonhydro
--------

This test represents the flow of dense fluid down a slope using the
nonhydrostatic model. After the creation of the mesh and initial conditions, 
a nonhydrostatic simulation is run for 30min. The ``namelist.forward`` file has 
namelist options related to run duration, time-stepping scheme, horizontal 
and vertical momentum viscosities, and defines the PETSc solver, 
preconditioner and tolerances used for the solution of the nonhydrostatic 
elliptic problem.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.overflow.nonhydro.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step.  A nonhydrostatic simulation is run on 8 cores 
for a run duration of 30min.

hydro_vs_nonhydro
-----------------

This test represents the flow of dense fluid down a slope and compares the 
solutions obtained with the hydrostatic and  nonhydrostatic model. After 
the creation of the mesh and initial conditions, an hydrostatic and a 
nonhydrostatic simulation are run. The ``namelist.forward`` file has a few 
common namelist options for the two models related to run duration, 
time-stepping scheme, and tracer advection. The files ``namelist.hydro`` and
``namelist.nonhydro`` specify the different momentum viscosities for the two
models, and the latter defines the PETSc solver, preconditioner and tolerances
used for the solution of the nonhydrostatic elliptic problem. The hydrostatic and
nonhydrostatic run share the same ``streams.forward`` file that defines
``mesh``, ``input``, ``restart``, and ``output`` streams.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.overflow.hydro_vs_nonhydro.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step.  The ``nonhydro_mode`` argument is a boolean that
determines if the hydrostatic or the nonhydrostatic model is run.
Namelist and streams files are generate during ``setup()`` and
MPAS-Ocean is run (including updating PIO namelist options and generating a
graph partition) in ``run()``. Both the hydrostatic and nonhydrostatic
simulation are run on 8 cores and have a run duration of 3h, the time at
which the dense fluid has descended the slope.

visualize
~~~~~~~~~

The ``visualize`` step defined by
:py:class:`compass.ocean.tests.overflow.hydro_vs_nonhydro.visualize.Visualize`
makes plots of the temperature profile at 3h for the hydrostatic
and nonhydrostatic case. The plot shows that a Kelvin-Helmholtz instability 
develops in the nonhydrostatic case, leading to entrainment of
ambient fluid into plumes, whereas the hydrostatic model fails to 
capture the correct physics.
