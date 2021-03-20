.. _dev_landice_greenland:

greenland
=========

The ``greenland`` configuration performs short (5-day) forward runs on a coarse
(20-km) Greenland mesh (see :ref:`landice_greenland`).  Here, we describe the
shared framework for this configuration and the 3 test cases.

.. _dev_landice_greenland_framework:

framework
---------

There are no shared configuration options for the ``greenland`` configuration.

The configuration has a shared ``namelist.landice`` file with
a few common namelist options related to time step, run duration and calving,
and a shared ``streams.landice`` file that defines ``input``, ``restart``, and
``output`` streams.

run_model
~~~~~~~~~

The module ``compass.landice.tests.greenland.run_model`` defines a step for
running MALI from the initial condition produced in the ``setup_mesh`` step.
For the ``restart_test`` test cases, the model will run multiple times with
different namelist and streams files.  To support this functionality, this step
has an entry in the ``step`` dictionary ``suffixes``, which is a list of
suffixes for the these namelist and streams files.  The model runs once for
each suffix.  The default is just ``landice``.

.. _dev_landice_greenland_smoke_test:

smoke_test
----------

This test performs a 5-day run on 4 cores.  It doesn't contain any
:ref:`dev_validation`.

.. _dev_landice_greenland_decomposition_test:

decomposition_test
------------------

This test performs a 5-day run once on 1 core and once on 4 cores.  It
ensures that ``thickness`` and ``normalVelocity`` are identical at the end of
the two runs (as well as with a baseline if one is provided when calling
:ref:`dev_compass_setup`).

.. _dev_landice_greenland_restart_test:

restart_test
------------

This test performs a 5-day run once on 4 cores, then a sequence of a 3-day and
a 2-day run on 4 cores.  It ensures that ``thickness`` and ``normalVelocity``
are identical at the end of the two runs (as well as with a baseline if one is
provided when calling :ref:`dev_compass_setup`).

The restart step works by creating two different namelist and streams files,
one each with ``landice`` as the suffix and one each with ``landice.rst`` as
the suffix.  The former perform a 3-day run from the initial condition, while
the latter perform a 2-day restart run beginning with the end of the first.
