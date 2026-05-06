.. _dev_ocean_ziso:

ziso
====

The ``ziso`` test group
(:py:class:`compass.ocean.tests.ziso.Ziso`)
implements variants of the Zonally Idealized Southern Ocean (ZISO) test case
(see :ref:`ocean_ziso` in the User's Guide) at 2.5, 5, 10 and 20-km
resolutions. Here, we describe the shared framework for this test group and the
4 types of test cases currently supported: ``default``, ``long``,
``particles``, and ``with_frazil``.

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
``initial_state`` step.  Namelist and streams files are generated when the test
case is set up (combining the ``*.forward`` and ``*.analysis`` files from the
test group) and updated at runtime based on the config options in the
``[ziso]`` section of the config file.

A dictionary ``res_params`` is used to set parameters like the number of cores
and the time steps for each resolution:

.. code-block:: python

    res_params = {'20km': {'cores': 20,
                           'min_tasks': 2,
                           'cores_with_particles': 32,
                           'min_cores_with_particles': 12,
                           'dt': "'00:12:00'",
                           'btr_dt': "'00:00:36'",
                           'mom_del4': "5.0e10",
                           'run_duration': "'0000_00:36:00'"},
                  '10km': {'cores': 80,
                           'min_tasks': 8,
                           'cores_with_particles': 130,
                           'min_cores_with_particles': 50,
                           'dt': "'00:06:00'",
                           'btr_dt': "'00:00:18'",
                           'mom_del4': "6.25e9",
                           'run_duration': "'0000_00:18:00'"},
                  '5km': {'cores': 300,
                          'min_tasks': 30,
                          'cores_with_particles': 500,
                          'min_cores_with_particles': 200,
                          'dt': "'00:03:00'",
                          'btr_dt': "'00:00:09'",
                          'mom_del4': "7.8e8",
                          'run_duration': "'0000_00:09:00'"},
                  '2.5km': {'cores': 1200,
                            'min_tasks': 120,
                            'cores_with_particles': 2100,
                            'min_cores_with_particles': 900,
                            'dt': "'00:01:30'",
                            'btr_dt': "'00:00:04'",
                            'mom_del4': "9.8e7",
                            'run_duration': "'0000_00:04:30'"}}

If :ref:`dev_ocean_framework_particles` are includes (in the ``particles``
test cases),  we first generate the required graph partition, then partition
the particles, and finally run MPAS-Ocean (including updating PIO namelist
options):

.. code-block:: python

    if self.with_particles:
        cores = self.cores
        partition(cores, self.config, self.logger)
        particles.write(init_filename='init.nc',
                        particle_filename='particles.nc',
                        graph_filename='graph.info.part.{}'.format(cores),
                        types='buoyancy')
        run_model(self, partition_graph=False)

.. _dev_ocean_ziso_default:

ziso_test_case
--------------

The :py:class:`compass.ocean.tests.ziso.ZisoTestCase` class defines most of the
ZISO test cases. If a baseline is provided when calling
:ref:`dev_compass_setup`, the test case ensures that the final values of
``temperature`` and ``layerThickness`` are identical to the baseline values.
If particles are included, a number of particle-related variables and timers
are also validated against the baseline.

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
