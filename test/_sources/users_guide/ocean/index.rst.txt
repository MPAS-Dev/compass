.. _ocean:

Ocean core
==========

The ``ocean`` core in compass contains test cases for the
`MPAS-Ocean <https://mpas-dev.github.io/ocean/ocean.html>`_) model.  For more
information on MPAS-Ocean, see the most recent
`user's guide <https://doi.org/10.5281/zenodo.1246893>`_ for version 6.
Currently, there are 4 configurations---:ref:`ocean_baroclinic_channel`,
:ref:`ocean_global_ocean`, :ref:`ocean_ice_shelf_2d`, and :ref:`ocean_ziso`,
each with several :ref:`test_cases`.  Many more ``ocean`` test cases are still
available through :ref:`legacy_compass` (see legacy
:ref:`legacy_compass_ocean_testcases`).

Some helpful external links:

* MPAS website: https://mpas-dev.github.io

* `MPAS-Ocean User's Guide <https://zenodo.org/record/1246893#.WvsFWNMvzMU>`_
  includes a quick start guide and description of all flags and variables.

* Test cases (not requiring ``compass``) for latest MPAS-Ocean release:

  * https://mpas-dev.github.io/ocean/releases.html

  * https://mpas-dev.github.io/ocean/release_6.0/release_6.0.html

.. toctree::
   :titlesonly:

   configurations/index
   framework/index
   suites
