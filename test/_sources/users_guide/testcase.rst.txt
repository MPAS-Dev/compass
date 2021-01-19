.. _test_cases:

Test Cases
==========

compass supports test cases for two main MPAS dynamical cores, :ref:`ocean`
(`MPAS-Ocean <https://mpas-dev.github.io/ocean/ocean.html>`_) and
:ref:`landice` (`MALI <https://mpas-dev.github.io/land_ice/land_ice.html>`_).
Test cases are grouped under these two "cores" and then into "configurations",
which are groups of test cases that are part of the same framework, serve a
similar purpose, or are variants on one another.  Example of ocean
configurations include "idealized" setups like :ref:`ocean_baroclinic_channel`,
:ref:`ocean_ziso` and "realistic" domains like :ref:`ocean_global_ocean`.
Idealized configurations typically use analytic functions to define their
topography, initial conditions and forcing data (i.e. boundary conditions),
whereas realistic configurations most often use data files for all fo these.

compass test cases are made up of one or more steps.  These are the smallest
units of work in compass. You can run an individual step on its own if you
like.  Currently, the steps in a test case run in sequence but there are plans
to allow steps that don't depend on one another to run in parallel in the
future.  Also, there is no requirement that all steps defined in a test case
must run when that test case is run.  Some steps may be disabled depending on
configuration options (see :ref:`config_files`) that you choose.

In compass, test cases are identified by their subdirectory relative to a work
directory that the user chooses.  For example, the default test case from
the :ref:`ocean_baroclinic_channel` configuration at 10-km resolution is
identified as:

.. code-block:: none

    ocean/baroclinic_channel/10km/default

When you list test cases:

.. code-block:: bash

    python -m compass list

you will see these relative paths.
