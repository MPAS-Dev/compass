.. _dev_ocean_merry_go_round:

merry_go_round
==============

The ``merry_go_round`` test group
(:py:class:`compass.ocean.tests.merry_go_round.MerryGoRound`)
implements variants of the merry-go-round test case.  Here,
we describe the shared framework for this test group and the 2 test cases.

.. _dev_ocean_merry_go_round_framework:

framework
---------

The shared config options for the ``merry_go_round`` test group
are described in :ref:`ocean_merry_go_round` in the User's Guide.

Additionally, the test group has a shared ``namelist.forward`` file with
a few common namelist options related to run duration and disabled tendency terms, as
well as a shared ``streams.forward`` file that defines ``mesh``, ``input``, ``restart``,
and ``output`` streams.

initial_state
~~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.merry_go_round.initial_state.InitialState`
defines a step for setting up the initial state for each test case.

First, a mesh appropriate for the resolution is generated using
:py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`.  Then, the mesh is
culled to remove periodicity in the x direction.  A vertical grid is generated,
with 50 layers of 10-m thickness each by default.  Finally, the initial
temperature, salinity, velocity, and debug tracer fields are computed.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.merry_go_round.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step.  If ``resolution`` is provided as an argument to the
constructor, it is used to modify the time step (``config_dt``). MPAS-Ocean is run
(including updating PIO namelist options and generating a graph partition) in ``run()``.

.. _dev_ocean_merry_go_round_default:

default
-------

The :py:class:`compass.ocean.tests.merry_go_round.default.Default` test performs a
6-hour run on 4 cores. The :ref:`dev_validation` is used to compare ``normalVelocity``
and ``tracer1`` variables with a baseline.

.. _dev_ocean_merry_go_round_convergence_test:

convergence_test
----------------

The :py:class:`compass.ocean.tests.merry_go_round.convergence_test.ConvergenceTest`
performs three 6-hour runs at three different resolutions (with concomittent refinement
of the time step). It doesn't contain any :ref:`dev_validation`.

The ``analysis`` step defined by
:py:class:`compass.ocean.tests.merry_go_round.covergence_test.analysis.Analysis`
makes a convegence plot of the root-mean-square-error of each run's final state with
respect to its initial state.

This test is resource intensive enough that it is not used in regression testing.
