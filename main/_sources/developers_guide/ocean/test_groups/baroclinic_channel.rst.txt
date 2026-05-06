.. _dev_ocean_baroclinic_channel:

baroclinic_channel
==================

The ``baroclinic_channel`` test group
(:py:class:`compass.ocean.tests.baroclinic_channel.BaroclinicChannel`)
implements variants of the Baroclinic Eddies test case
(see :ref:`ocean_baroclinic_channel`) at 3 resolutions (1, 4 and 10 km).  Here,
we describe the shared framework for this test group and the 5 test cases.

.. _dev_ocean_baroclinic_channel_framework:

framework
---------

The shared config options for the ``baroclinic_channel`` test group
are described in :ref:`ocean_baroclinic_channel` in the User's Guide.

Additionally, the test group has a shared ``namelist.forward`` file with
a few common namelist options related to run duration and default horizontal
and vertical momentum and tracer diffusion, as well as a shared
``streams.forward`` file that defines ``mesh``, ``input``, ``restart``, and
``output`` streams.  There are also ``namelist.forward.<res>`` files for each
resolution of the ``rpe_test`` test case, which define the horizontal
viscosity and time steps.

initial_state
~~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.baroclinic_channel.initial_state.InitialState`
defines a step for setting up the initial state for each test case.

First, a mesh appropriate for the resolution is generated using
:py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`.  Then, the mesh is
culled to remove periodicity in the y direction.  A vertical grid is generated,
with 20 layers of 50-m thickness each by default.  Finally, the initial
temperature field is computed along with uniform salinity and zero initial
velocity.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.baroclinic_channel.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step.  If ``nu`` is provided as an argument to the
constructor, the associate namelist option (``config_mom_del2``) will be given
this value. Namelist and streams files are generate during ``setup()`` and
MPAS-Ocean is run (including updating PIO namelist options and generating a
graph partition) in ``run()``.

.. _dev_ocean_baroclinic_channel_default:

default
-------

The :py:class:`compass.ocean.tests.baroclinic_channel.default.Default`
test performs a 15-minute run on 4 cores.  It doesn't contain any
:ref:`dev_validation`.

.. _dev_ocean_baroclinic_channel_decomp_test:

decomp_test
-----------

The :py:class:`compass.ocean.tests.baroclinic_channel.decomp_test.DecompTest`
performs a 15-minute run once on 4 cores and once on 8 cores.  It
ensures that ``temperature``, ``salinity``, ``layerThickness`` and
``normalVelocity`` are identical at the end of the two runs (as well as with a
baseline if one is provided when calling :ref:`dev_compass_setup`).

.. _dev_ocean_baroclinic_channel_thread_test:

threads_test
------------

The :py:class:`compass.ocean.tests.baroclinic_channel.threads_test.ThreadsTest`
performs a 15-minute run once on 4 cores, each with 1 thread and once on 4
cores, each with 2 threads.  It ensures that ``temperature``, ``salinity``,
``layerThickness`` and ``normalVelocity`` are identical at the end of the two
runs (as well as with a baseline if one is provided when calling
:ref:`dev_compass_setup`).

.. _dev_ocean_baroclinic_channel_restart_test:

restart_test
------------

The :py:class:`compass.ocean.tests.baroclinic_channel.restart_test.RestartTest`
performs a 10-minute run once on 4 cores, saving restart files every time step
(every 5 minutes), then it performs a restart run starting at minute 5 for 5
more minutes.  It ensures that ``temperature``, ``salinity``,
``layerThickness`` and ``normalVelocity`` are identical at the end of the two
runs (as well as with a baseline if one is provided when calling
:ref:`dev_compass_setup`).

Restart files are saved at the test-case level in the ``restarts`` directory,
rather than within each step, since they will be used across both the ``full``
and ``restart`` steps.

The ``namelist.full`` and ``streams.full`` files are used to set up the run
duration and restart frequency of the full run, while ``namelist.restart`` and
``streams.restart`` make sure that the restart step begins with a restart at
minute 5 and runs for 5 more minutes.

.. _dev_ocean_baroclinic_channel_rpe_test:

rpe_test
--------

The :py:class:`compass.ocean.tests.baroclinic_channel.rpe_test.RpeTest`
performs a longer (20 day) integration of the model forward in time at 5
different values of the viscosity.  Versions of the test case exist at each of
the 3 supported horizontal resolutions (1, 4 and 10 km).

The different resolutions use different numbers of resources, as determined by
a python dictionary:

.. code-block:: python

    res_params = {'1km': {'ntasks': 144, 'min_tasks': 36},
                  '4km': {'ntasks': 36, 'min_tasks': 8},
                  '10km': {'ntasks': 8, 'min_tasks': 4}}

    if resolution not in res_params:
        raise ValueError(
            f'Unsupported resolution {resolution}. Supported values are: '
            f'{list(res_params)}')

    params = res_params[resolution]

These ``params`` are later added as arguments to the ``Forward`` steps'
constructors when they are added to the test case:

.. code-block:: python

        step = Forward(
            test_case=self, name=name, subdir=name,
            ntasks=params['ntasks'], min_tasks=params['min_tasks'],
            resolution=resolution, nu=float(nu))
        ...
        self.add_step(step)

The ``analysis`` step defined by
:py:class:`compass.ocean.tests.baroclinic_channel.rpe_test.analysis.Analysis`
makes plots of the final results with each value of the viscosity.

This test is resource intensive enough that it is not used in regression
testing.
