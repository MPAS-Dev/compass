.. _dev_landice_thwaites:

thwaites
=========

The ``thwaites`` test group (:py:class:`compass.landice.tests.thwaites.Thwaites`)
performs short (5-day) forward runs on a coarse (4-14 km) thwaites mesh
(see :ref:`landice_thwaites`).  Here, we describe the shared framework for
this test group and the 2 test cases.

.. _dev_landice_thwaites_framework:

framework
---------

There are no shared config options for the ``thwaites`` test group.

The Thwaites test can only be run with the FO velocity solver.
Running with the FO solver requires building MALI with the Albany
library.

The test group has a shared ``namelist.landice`` file with
a few common namelist options related to time step, run duration and calving,
and a shared ``streams.landice`` file that defines ``input``, ``restart``, and
``output`` streams.

run_model
~~~~~~~~~

The class :py:class:`compass.landice.tests.thwaites.run_model.RunModel`
defines a step for running MALI an initial condition downloaded from
`thwaites.4km.210608.nc <https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/mpas-albany-landice/thwaites.4km.210608.nc>`_.
For the ``restart_test`` test cases, the model will run multiple times with
different namelist and streams files.  To support this functionality, this step
has an attribute ``suffixes``, which is a list of suffixes for the these
namelist and streams files.  The model runs once for each suffix.  The default
is just ``landice``.

.. _dev_landice_thwaites_decomposition_test:

decomposition_test
------------------

The :py:class:`compass.landice.tests.thwaites.decomposition_test.DecompositionTest`
performs a 5-day run once on 1 core and once on 4 cores.  It ensures that
``thickness`` and ``surfaceSpeed`` are identical at the end of the two runs
(as well as with a baseline if one is provided when calling
:ref:`dev_compass_setup`).

.. _dev_landice_thwaites_restart_test:

restart_test
------------

The :py:class:`compass.landice.tests.thwaites.restart_test.RestartTest`
performs a 5-day run once on 4 cores, then a sequence of a 3-day and a 2-day
run on 4 cores.  It ensures that ``thickness`` and ``surfaceSpeed`` are
identical at the end of the two runs (as well as with a baseline if one is
provided when calling :ref:`dev_compass_setup`).

The restart step works by creating two different namelist and streams files,
one each with ``landice`` as the suffix and one each with ``landice.rst`` as
the suffix.  The former perform a 3-day run from the initial condition, while
the latter perform a 2-day restart run beginning with the end of the first.

mesh
~~~~

The class :py:class:`compass.landice.tests.thwaites.mesh.Mesh`
defines a step for creating a variable resolution Thwaites Glacier mesh.
This is used by the ``mesh_gen`` test case.

mesh_gen
-------------

The :py:class:`compass.landice.tests.thwaites.mesh_gen.MeshGen`
calls the :py:class:`compass.landice.tests.thwaites.mesh.Mesh` to create
the variable resolution Thwaites Glacier mesh.
