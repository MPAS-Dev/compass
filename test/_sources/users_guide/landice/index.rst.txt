.. _landice:

Landice core
============

The ``landice`` core in compass contains test cases for the MPAS-Albany
Land Ice (`MALI <https://mpas-dev.github.io/land_ice/land_ice.html>`_) model.
For more information on MALI, see the most recent
`Hoffman et al. (2018) <https://doi.org/10.5194/gmd-11-3747-2018>`_.
Currently, there are 2 configurations---:ref:`landice_dome` and
:ref:`landice_enthalpy_benchmark`, each with several :ref:`test_cases`.  Many
more ``landice`` test cases are still available through :ref:`legacy_compass`.

Some helpful external links:

* MPAS website: https://mpas-dev.github.io

* `MPAS-Albany Land Ice Model User's Guide v6.0 <https://doi.org/10.5281/zenodo.1227426>`_
  includes a quick start guide and description of all flags and variables.

* Test cases (not requiring ``compass``) for latest MPAS-Ocean release:

  * https://mpas-dev.github.io/land_ice/download.html

  * `MPAS-Albany Land Ice v6.0 Test Cases <https://doi.org/10.5281/zenodo.1227430>`_

.. toctree::
   :titlesonly:

   configurations/index
   framework/index
   suites