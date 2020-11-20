.. _compass_ocean:


Introduction
============

Some helpful external links:

* MPAS website: https://mpas-dev.github.io

* `MPAS-Ocean User's Guide <https://zenodo.org/record/1246893#.WvsFWNMvzMU>`_
  includes a quick start guide and description of all flags and variables.

* Test cases for latest release version:

  * https://mpas-dev.github.io/ocean/releases.html

  * https://mpas-dev.github.io/ocean/release_6.0/release_6.0.html

.. _setup_ocean:

Setting config options for ocean test cases
-------------------------------------------

This documentation is similar to :ref:`setup_overview`, with some details that
are specific to the ``ocean`` core. If you are new to MPAS-Ocean, it is easiest
to `download a prepared test case <https://mpas-dev.github.io/ocean/release_6.0/release_6.0.html>`_.

If you want to set up a test case with COMPASS, you first need to set some
config options. The file ``general.config.ocean`` is a template showing which
paths need to bet set in order to set up ocean test cases. If you are using
LANL IC (Grizzly or Badger) or NERSC (Cori), you can use these example files,
which give the paths `mark-petersen's <https://github.com/mark-petersen>`_ uses:

* LANL IC: `config.ocean_LANL_IC <https://gist.github.com/mark-petersen/4e4fd40407c2a326ce286ab6b81f44fb>`_

* NERSC: `general.config.ocean_cori <https://gist.github.com/mark-petersen/c61095d65216415ee0bb62a76da3c6cb>`_

You should change the MPAS repo to your directories to test your own code.

Alternatively, you can make a copy of this file (e.g. ``config.ocean``) and set
the options as follows. In six places, replace ``FULL_PATH_TO_MPAS_MODEL_REPO``
with the path where you have checked out (and built) the branch of MPAS-Model
you are planning to use.

Three other paths are required, ``mesh_database``,
``initial_condition_database`` and ``bathymetry_database``, are used for
storing pre-generated mesh files, data sets for creating initial conditions,
and bathymetry data. These can be empty directories, in which case meshes and
other data sets will be downloaded as required during test-case setup.  (If a
test case appears to hang during setup, it is most likely downloading mesh, 
initial-condition or bathymetry data.)

On LANL IC, the shared data bases can be found at:

.. code-block:: ini

    mesh_database = /usr/projects/regionalclimate/COMMON_MPAS/ocean/grids/mesh_database
    initial_condition_database = /usr/projects/regionalclimate/COMMON_MPAS/ocean/grids/initial_condition_database
    bathymetry_database = /usr/projects/regionalclimate/COMMON_MPAS/ocean/grids/bathymetry_database

On NERSC (Cori), the shared data bases can be found at:

.. code-block:: ini

    mesh_database = /project/projectdirs/e3sm/mpas_standalonedata/mpas-ocean/mesh_database/
    initial_condition_database = /project/projectdirs/e3sm/mpas_standalonedata/mpas-ocean/initial_condition_database/
    bathymetry_database = /project/projectdirs/e3sm/mpas_standalonedata/mpas-ocean/bathymetry_database/

