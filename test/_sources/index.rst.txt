compass
=======

Configuration Of Model for Prediction Across Scales Setups (compass) is an
automated system to set up test cases that remain synchronized with the
`MPAS-Model repository <https://github.com/MPAS-Dev/MPAS-Model>`_.

Many compass test cases are idealized, and are used for things like
performing convergence tests or regression tests on particular parts of the
MPAS code.  Many compass test cases, such as those under the
:ref:`ocean_global_ocean` configuration, are "realistic" in the sense that they
use data sets from observations to create create global and regional meshes,
initial conditions, and boundary conditions.

``compass`` will be the tool used to create new land-ice and ocean meshes and
initial conditions for future versions of `E3SM <https://e3sm.org/>`_, just as
:ref:`legacy_compass` has been used to create meshes and initial conditons for
`E3SM v1 <https://e3sm.org/model/e3sm-model-description/v1-description/>`_
and `v2 <https://e3sm.org/research/science-campaigns/v2-planned-campaign/>`_.
We note that ``compass`` does *not* provide the tools for creating many of the
files needed for full E3SM coupling, a process that requires expert help from
the E3SM development team.

The ``compass`` python package defines the test cases along with the commands
to list and set up both test cases and test suites (groups of test cases).
compass currently supports the ``landice`` and ``ocean`` dynamical cores for
MPAS.  Nearly all test cases include calls that launch one of these dynamical
cores.  These runs are configured with namelists and streams files, and one of
the benefits of using compass over attempting to run one of the MPAS components
directly is that compass begins with default values for all namelists and
streams, modifying only those options where the default is not appropriate.
In this way, compass requires little alteration as MPAS itself evolves and new
functionality is added.

.. toctree::
   :caption: User's guide
   :maxdepth: 2

   users_guide/overview
   users_guide/testcase
   users_guide/configuration
   users_guide/suite
   users_guide/landice/index
   users_guide/ocean/index
   users_guide/machines/index

.. toctree::
   :caption: Developer's guide
   :maxdepth: 2

   developers_guide/overview
   developers_guide/building_docs
   developers_guide/api

.. _legacy_compass:

Legacy COMPASS
==============

The legacy version of COMPASS is controlled by 4 scripts
(:ref:`legacy_list_testcases`, :ref:`legacy_setup_testcase`,
:ref:`legacy_clean_testcase`, and :ref:`legacy_manage_regression_suite`), and
test cases are defined by a series of XML files with a strict directory
structure.  Over the years, we have found this structure to be confusing for
both new and experienced developers, with serious limitations on flexibility
of test case design and code reuse.  We do not anticipate any new test cases
being added to the legacy version but it will take time for legacy test cases
to be ported to the new ``compass`` python package.  Therefore, the legacy
framework and test cases are expected to persist for the medium term.

.. toctree::
   :caption: Legacy User's guide
   :maxdepth: 2

   legacy/users_guide/overview
   legacy/users_guide/scripts

.. toctree::
   :caption: Legacy Ocean Test Cases
   :maxdepth: 2

   legacy/ocean/introduction
   legacy/ocean/test_cases/index

.. toctree::
   :caption: Legacy Instructions for Machines
   :maxdepth: 2

   legacy/machine_specific_instructions/slurm
   legacy/machine_specific_instructions/lanl
   legacy/machine_specific_instructions/nersc
   legacy/machine_specific_instructions/anvil
   legacy/machine_specific_instructions/linux
   legacy/machine_specific_instructions/osx

.. toctree::
   :caption: Legacy Developer's guide
   :maxdepth: 2

   legacy/developers_guide/config
   legacy/developers_guide/driver_script
   legacy/developers_guide/template
   legacy/developers_guide/regression_suite
   legacy/developers_guide/run_config
   legacy/developers_guide/building_docs
