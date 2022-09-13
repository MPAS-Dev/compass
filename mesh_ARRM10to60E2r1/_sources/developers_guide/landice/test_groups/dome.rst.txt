.. _dev_landice_dome:

dome
====

The ``dome`` test group (:py:class:`compass.landice.tests.dome.Dome`)
implements variants of the dome test case either at 2000-m uniform or variable
horizontal resolution and 10 vertical layers (see :ref:`landice_dome`). Here,
we describe the shared framework for this test group and the 3 test cases
implemented for each resolution.

.. _dev_landice_dome_framework:

framework
---------

The shared config options for the ``dome`` test group are described
in :ref:`landice_dome` in the User's Guide.

Additionally, the test group has a shared ``namelist.landice`` file with
a few common namelist options related to time step and run duration, and a
shared ``streams.landice`` file that defines ``input``, ``restart``, and
``output`` streams.

SetupMesh
~~~~~~~~~

The class :py:class:`compass.landice.tests.dome.setup_mesh.SetupMesh` defines a
step for setting up the mesh for each test case.

For test cases at uniform, 2000-m resolution, the horizontal mesh is
constructed at runtime (using
:py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`).
The variable resolution mesh is downloaded from
`dome_varres_grid.nc <https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/mpas-albany-landice/dome_varres_grid.nc>`_:

.. code-block:: python

    self.add_input_file(filename='mpas_grid.nc',
                        target='dome_varres_grid.nc', database='')

.. note::

    In this case, we use ``database=''`` as a trick to download the file from
    the LCRC server but without any subdirectory relative to the path pointed
    to by the ``server_base_url`` and ``core_path`` options in the ``download``
    section of the config file.

A MALI grid is created with the MPAS-Tools script
``create_landice_grid_from_generic_MPAS_grid.py`` and a graph file is created
to partition the mesh before the model run.

Finally, the initial condition is defined in the private function
``_setup_dome_initial_conditions()``.

RunModel
~~~~~~~~

The class :py:class:`compass.landice.tests.dome.run_model.RunModel` defines a
step for running MALI from the initial condition produced in ``SetupMesh``.
For the ``RestartTest`` test cases, the model will run multiple times with
different namelist and streams files.  To support this functionality, this
constructor takes an argument ``suffixes``, which is a list of suffixes for the
these namelist and streams files.  The model runs once for each suffix.  The
default is just ``landice``.

Visualize
~~~~~~~~~

The :py:class:`compass.landice.tests.dome.visualize.Visualize` step is optional
in each test case and can be run manually to plot the results of the test case.
It is control by the config options in the ``dome_viz`` section.

This step is not run by default by adding it with ``run_by_default=False``:

.. code-block:: python

    step = Visualize(test_case=self, mesh_type=mesh_type)
    self.add_step(step, run_by_default=False)

.. _dev_landice_dome_smoke_test:

SmokeTest
---------

The :py:class:`compass.landice.tests.dome.smoke_test.SmokeTest` performs a
200-year run on 4 cores.  It doesn't contain any :ref:`dev_validation`.

.. _dev_landice_dome_decomposition_test:

DecompositionTest
-----------------

The :py:class:`compass.landice.tests.dome.decomposition_test.DecompositionTest`
performs a 200-year run once on 1 core and once on 4 cores.  It ensures that
``thickness`` and ``normalVelocity`` are identical at the end of the two runs
(as well as with a baseline if one is provided when calling
:ref:`dev_compass_setup`).

.. _dev_landice_dome_restart_test:

RestartTest
-----------

The :py:class:`compass.landice.tests.dome.restart_test.RestartTest` performs a
2-year run once on 4 cores, then a sequence of 2 1-year runs on 4 cores.  It
ensures that ``thickness`` and ``normalVelocity`` are identical at the end of
the two runs (as well as with a baseline if one is provided when calling
:ref:`dev_compass_setup`).

The restart step works by creating two different namelist and streams files,
one each with ``landice`` as the suffix and one each with ``landice.rst`` as
the suffix.  The former perform a 1-year run from the initial condition, while
the latter perform a 1-year restart run beginning with the end of the first.
