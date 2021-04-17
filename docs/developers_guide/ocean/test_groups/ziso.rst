.. _dev_ocean_ziso:

ziso
====

The ``ziso`` test group
(:py:class:`compass.ocean.tests.ziso.Ziso`)
implements variants of the Zonally Idealized Southern Ocean (ZISO) test case
(see :ref:`ocean_ziso`) at 20-km resolutions. Here, we describe the shared
framework for this test group and the 2 test cases.

.. _dev_ocean_ziso_framework:

framework
---------

The shared config options for the ``ziso`` test group are described
in :ref:`ocean_ziso` in the User's Guide.

Additionally, the test group has several shared namelist and streams files,
some for shared parameters and streams for forward runs (``namelist.forward``
and ``streams.forward``), some specific to the resolution of the run and
with an eye toward including more resolutions in the future
(``namelist.<res>.forward`` and ``streams.<res>.forward``), and some related
to enabling analysis members in a run and outputting their results
(``namelist.analysis`` and ``streams.analysis``).

initial_state
~~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.ziso.initial_state.InitialState`
defines a step for setting up the initial state for each test case.

First, a mesh appropriate for the resolution is generated using
:py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`.  Then, the mesh is
culled to remove periodicity in the y direction.  A vertical grid is generated,
with 100 non-uniform layers according to the ``100layerE3SMv1`` distribution,
but squashed so the bottom of the deepest layer is at 2500 m depth.  Finally,
the initial temperature field is computed along with uniform salinity and
zero initial velocity.  The temperature profile is significantly different
for test cases with frazil-ice production compared to those without.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.ziso.forward.Forward` defines a step
for running MPAS-Ocean from the initial condition produced in the
``initial_state`` step.  If ``with_frazil = True``, frazil ice production is
enabled; if ``with_analysis = True``, analysis members are enabled:

.. code-block:: python

    if with_analysis:
        add_namelist_file(step, 'compass.ocean.tests.ziso',
                          'namelist.analysis')
        add_streams_file(step, 'compass.ocean.tests.ziso', 'streams.analysis')

    if with_frazil:
        add_namelist_options(step,
                             {'config_use_frazil_ice_formation': '.true.'})
        add_streams_file(step, 'compass.ocean.streams', 'streams.frazil')

Namelist and streams files are generate during ``setup()``.

:ref:`dev_ocean_framework_particles` are included in all simulations.  In order
to partition the particles, we need to first generate the required graph
partition, then partition the particles, and finally run MPAS-Ocean (including
updating PIO namelist options):

.. code-block:: python

    cores = step['cores']

    partition(cores, config, logger)
    particles.write(init_filename='init.nc', particle_filename='particles.nc',
                    graph_filename='graph.info.part.{}'.format(cores),
                    types='buoyancy')
    run_model(step, config, logger, partition_graph=False)

.. _dev_ocean_ziso_default:

default
-------

The :py:class:`compass.ocean.tests.ziso.default.Default` test performs a
90-second (3-time-step) run on 4 cores, including analysis but without
frazil-ice formation. If a baseline is provided when calling
:ref:`dev_compass_setup`, the test case ensures that the final values of
``temperature`` and ``layerThickness`` as well as a number of particle-related
variables are identical to the baseline values, and also compares timers
related to particles with the baseline.

.. _dev_ocean_ziso_with_frazil:

with_frazil
-----------

The :py:class:`compass.ocean.tests.ziso.with_frazil.WithFrazil`
includes default config options:

.. code-block:: cfg

    # namelist options for Zonally periodic Idealized Southern Ocean (ZISO)
    # testcases
    [ziso]

    # Initial temperature profile constant
    initial_temp_t1 = 0.0

    # Initial temperature profile tanh coefficient
    initial_temp_t2 = -1.0

    # Initial temperature profile tanh length scale
    initial_temp_h1 = 300.0

    # Initial temperature profile linear coefficient
    initial_temp_mt = 0.0

This test performs a 90-second (3-time-step) run on 4 cores, including
frazil-ice formation but without analysis. If a baseline is provided when
calling :ref:`dev_compass_setup`, the test case ensures that the final values
of ``temperature`` and ``layerThickness`` as well as a number of frazil-related
variables are identical to the baseline values.
