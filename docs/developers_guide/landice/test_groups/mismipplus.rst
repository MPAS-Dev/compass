.. _dev_landice_mismipplus:

mismipplus
==========

The ``mismipplus`` test group (:py:class:`compass.landice.tests.mismipplus.MISMIPplus`)
performs a short smoke test of the MISMIP+ mesh
(see :ref:`landice_mismipplus`).  Here, we describe the shared framework for
this test group and the 1 test case.

.. _dev_landice_mismipplus_framework:

framework
---------

There are no shared config options for the ``mismipplus`` test group.

The mismipplus test should only be run with the FO velocity solver,
or with a data velocity field.
Running with the FO solver requires building MALI with the Albany
library.

The test group has a shared ``namelist.landice`` file with
a few common namelist options related to time step and run duration,
and a shared ``streams.landice`` file that defines ``input``, ``restart``, and
``output`` streams.
There also is an ``albany_input.yaml`` file for running the FO solver.

run_model
~~~~~~~~~

The class :py:class:`compass.landice.tests.mismipplus.run_model.RunModel`
defines a step for running MALI using an initial condition downloaded from
the shared data server.

smoke_test
----------

The :py:class:`compass.landice.tests.mismipplus.smoke_test.SmokeTest`
performs a 5 year run.  There is a validation step that compares the output
file against itself.  This is to allow the test to be compared against a
baseline if desired.
