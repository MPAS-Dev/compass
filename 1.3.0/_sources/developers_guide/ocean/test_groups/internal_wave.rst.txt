.. _dev_ocean_internal_wave:

internal_wave
=============

The ``internal_wave`` test group
(:py:class:`compass.ocean.tests.internal_wave.InternalWave`)
implements variants of the internal wave test case.  Here,
we describe the shared framework for this test group and the 3 test cases.

.. _dev_ocean_internal_wave_framework:

framework
---------

The shared config options for the ``internal_wave`` test group
are described in :ref:`ocean_internal_wave` in the User's Guide.

Additionally, the test group has a shared ``namelist.forward`` file with
a few common namelist options related to run duration and default horizontal
and vertical momentum and tracer diffusion, as well as a shared
``streams.forward`` file that defines ``mesh``, ``input``, ``restart``, and
``output`` streams.

initial_state
~~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.internal_wave.initial_state.InitialState`
defines a step for setting up the initial state for each test case.

First, a mesh appropriate for the resolution is generated using
:py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`.  Then, the mesh is
culled to remove periodicity in the y direction.  A vertical grid is generated,
with 20 layers of 25-m thickness each by default.  Finally, the initial
temperature field is computed along with uniform salinity and zero initial
velocity.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.internal_wave.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step.  If ``nu`` is provided as an argument to the
constructor, the associate namelist option (``config_mom_del2``) will be given
this value. If ``vlr`` is true, ``config_vertical_advection_method`` will be
assigned ``"remap"`` rather than the default value of ``"flux-form"``.
Namelist and streams files are generate during ``setup()`` and MPAS-Ocean is
run (including updating PIO namelist options and generating a graph partition)
in ``run()``.

.. _dev_ocean_internal_wave_default:

default
-------

The :py:class:`compass.ocean.tests.internal_wave.default.Default`
test performs a 15-minute run on 4 cores.  It doesn't contain any
:ref:`dev_validation`.

There is also a variant for testing vertical Lagrangian remap capabilities at
``ocean/internal_wave/vlr/default``.

.. _dev_ocean_internal_wave_rpe_test:

rpe_test
--------

The :py:class:`compass.ocean.tests.internal_wave.rpe_test.RpeTest`
performs a longer (20 day) integration of the model forward in time at 5
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
:py:class:`compass.ocean.tests.internal_wave.rpe_test.analysis.Analysis`
makes plots of the final results with each value of the viscosity.

This test is resource intensive enough that it is not used in regression
testing.

There is also a variant for testing vertical Lagrangian remap capabilities at
``ocean/internal_wave/vlr/rpe_test``.

.. _dev_ocean_internal_wave_ten_day_test:

ten_day_test
------------

The :py:class:`compass.ocean.tests.internal_wave.ten_day_test.TenDayTest`
performs a longer (10 day) integration of the model forward in time.
