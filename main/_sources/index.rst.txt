Compass
=======

Configuration Of Model for Prediction Across Scales Setups (Compass) is
a python package that provides an automated system to set up test cases for
Model for Prediction Across Scales (MPAS) components.  The development version
of Compass will be kept closely synchronized with the
`E3SM repo <https://github.com/E3SM-Project/E3SM>`_ and
`MALI-Dev repo <https://github.com/MALI-Dev/E3SM>`_. Release
versions will be compatible with specific tags of the MPAS components.

Many compass test cases are idealized, and are used for things like
performing convergence tests or regression tests on particular parts of the
MPAS code.  Many compass test cases, such as those under the
:ref:`ocean_global_ocean` test group, are "realistic" in the sense that they
use data sets from observations to create create global and regional meshes,
initial conditions, and boundary conditions.

Compass will be the tool used to create new land-ice and ocean meshes and
initial conditions for future versions of `E3SM <https://e3sm.org/>`_, just as
:ref:`legacy_compass` has been used to create meshes and initial conditions for
`E3SM v1 <https://e3sm.org/model/e3sm-model-description/v1-description/>`_
and `v2 <https://e3sm.org/research/science-campaigns/v2-planned-campaign/>`_.
We note that Compass does *not* provide the tools for creating many of the
files needed for full E3SM coupling, a process that requires expert help from
the E3SM development team.

The ``compass`` python package defines the test cases along with the commands
to list and set up both test cases and test suites (groups of test cases).
Compass currently supports the ``landice`` and ``ocean`` dynamical cores
for MPAS.  Nearly all test cases include calls that launch one of these
dynamical cores.  These runs are configured with namelists and streams files,
and one of the benefits of using Compass over attempting to run one of the
MPAS components directly is that Compass begins with default values for all
namelists and streams, modifying only those options where the default is not
appropriate. In this way, compass requires little alteration as the MPAS
components themselves evolves and new functionality is added.

Compass makes extensive use of the
`Jigsaw <https://github.com/dengwirda/jigsaw>`_ and
`Jigsaw-Python <https://github.com/dengwirda/jigsaw-python>`_ tools to make all
but the simplest meshes for our test cases and E3SM initial conditions.  These
tools, without which Compass' mesh generation capabilities
would not be possible, are developed primarily by
`Darren Engwirda <https://dengwirda.github.io/>`_.

.. toctree::
   :caption: User's guide
   :maxdepth: 2

   users_guide/quick_start
   users_guide/test_cases
   users_guide/config_files
   users_guide/test_suites
   users_guide/landice/index
   users_guide/ocean/index
   users_guide/machines/index

.. toctree::
   :caption: Developer's guide
   :maxdepth: 2

   developers_guide/quick_start
   developers_guide/overview
   developers_guide/command_line
   developers_guide/organization
   developers_guide/landice/index
   developers_guide/ocean/index
   developers_guide/framework
   developers_guide/machines/index
   developers_guide/troubleshooting
   developers_guide/docs
   developers_guide/building_docs
   developers_guide/deploying_spack
   developers_guide/api

   design_docs/index

.. toctree::
   :caption: Tutorials
   :maxdepth: 1

   tutorials/dev_add_test_group
   tutorials/dev_add_rrm
   tutorials/dev_add_param_study
   tutorials/dev_porting_legacy

.. toctree::
   :caption: Glossary
   :maxdepth: 2

   glossary

.. toctree::
   :caption: Versions
   :maxdepth: 2

   versions

.. _legacy_compass:

Legacy COMPASS
==============

The legacy version of COMPASS is controlled by 4 scripts
(`list_testcases.py <https://mpas-dev.github.io/compass/legacy/users_guide/scripts.html#list-testcases-py>`_,
`setup_testcases.py <https://mpas-dev.github.io/compass/legacy/users_guide/scripts.html#setup-testcases-py>`_,
`clean_testcases.py <https://mpas-dev.github.io/compass/legacy/users_guide/scripts.html#clean-testcases-py>`_,
and
`manage_regression_suite.py <https://mpas-dev.github.io/compass/legacy/users_guide/scripts.html#manage-regression-suite-py>`_),
and test cases are defined by a series of XML files with a strict directory
structure.  Over the years, we have found this structure to be confusing for
both new and experienced developers, with serious limitations on flexibility
of test case design and code reuse.  We do not anticipate any new test cases
being added to the legacy version but it will take time for legacy test cases
to be ported to the new ``compass`` python package.  Therefore, the legacy
framework and test cases are expected to persist for the medium term.

Documentation for legacy COMPASS can be found at:

http://mpas-dev.github.io/compass/legacy/

the code can be found in the ``legacy`` branch on the COMPASS GitHub repo:

https://github.com/MPAS-Dev/compass/tree/legacy
