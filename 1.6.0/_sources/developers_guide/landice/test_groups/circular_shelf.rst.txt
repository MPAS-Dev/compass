.. _dev_landice_circular_shelf:

circular_shelf
==============

The ``circular_shelf`` test group (:py:class:`compass.landice.tests.circular_shelf.CircularShelf`)
implements variants a circular_shelf test case (see :ref:`landice_circular_shelf`). Here,
we describe the shared framework for this test group.

.. _dev_landice_circular_shelf_framework:

framework
---------

The shared config options for the ``circular_shelf`` test group are described
in :ref:`landice_circular_shelf` in the User's Guide.

Additionally, the test group has a shared ``namelist.landice`` file with
appropriate namelist options, and a ``streams.landice`` file that defines
``input``, ``restart``, and ``output`` streams.

SetupMesh
~~~~~~~~~

The class :py:class:`compass.landice.tests.circular_shelf.setup_mesh.SetupMesh` defines a
step for setting up the mesh for the test case.

The horizontal mesh is
constructed at runtime (using
:py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`).

A MALI grid is created with the MPAS-Tools script
``create_landice_grid_from_generic_MPAS_grid.py`` and a graph file is created
to partition the mesh before the model run.

Finally, the initial condition is defined in the private function
``_setup_circular_shelf_initial_conditions()``.

RunModel
~~~~~~~~

The class :py:class:`compass.landice.tests.circular_shelf.run_model.RunModel` defines a
step for running MALI from the initial condition produced in ``SetupMesh``.

Visualize
~~~~~~~~~

The :py:class:`compass.landice.tests.circular_shelf.visualize.Visualize` step is optional
in each test case and can be run manually to plot the results of the test case.
It is control by the config options in the ``circular_shelf_viz`` section.

This step is not run by default by adding it with ``run_by_default=False``:

.. code-block:: python

    step = Visualize(test_case=self, mesh_type=mesh_type)
    self.add_step(step, run_by_default=False)

.. _dev_landice_circular_shelf_smoke_test:

DecompositionTest
-----------------

The :py:class:`compass.landice.tests.circular_shelf.decomposition_test.DecompositionTest`
performs a 200-year run once on 1 core and once on 4 cores.  It ensures that
``normalVelocity``, ``uReconstructX``, and ``uReconstructY`` are within a
small tolerance of each other between the two runs (as well as with a baseline
if one is provided when calling :ref:`dev_compass_setup`).

.. _dev_landice_circular_shelf_restart_test:
