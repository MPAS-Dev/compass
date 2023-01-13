.. _dev_ocean_overflow:

overflow
========

The ``overflow`` test group
(:py:class:`compass.ocean.tests.overflow.Overflow`)
implements variants of the continental shelf overflow test case.  Here,
we describe the shared framework for this test group and the 2 test cases.

.. _dev_ocean_overflow_framework:

framework
---------

The shared config options for the ``overflow`` test group
are described in :ref:`ocean_overflow` in the User's Guide.

Additionally, the test group has a shared ``namelist.forward`` file with
a few common namelist options related to time step, run duration, viscosity,
and drag, as well as a shared ``streams.forward`` file that defines ``mesh``,
``input``, and ``output`` streams.

initial_state
~~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.overflow.initial_state.InitialState`
defines a step for setting up the initial state for each test case.

First, a mesh appropriate for the resolution is generated using
:py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`.  Then, the mesh is
culled to remove periodicity in the y direction. The ocean model is then run
in ``init`` mode to generate the vertical grid and populate initial conditions.
 
forward
~~~~~~~

The class :py:class:`compass.ocean.tests.overflow.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step. If ``nu`` is provided as an argument to the
constructor, the associate namelist option (``config_mom_del2``) will be given
this value. Namelist and streams files are also generated. MPAS-Ocean is run
(including updating PIO namelist options and generating a graph partition) in
``run()``.

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
