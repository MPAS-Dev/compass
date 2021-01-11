COMPASS
=======

Configuration Of Model for Prediction Across Scales Setups (COMPASS) is an
automated system to set up test cases that remain synchronized with the
`MPAS-Model repository <https://github.com/MPAS-Dev/MPAS-Model>`_. The
``compass`` python package defines the test cases along with the commands to
list and set up both test cases and test suites (groups of test cases).
COMPASS currently supports the ``landice`` and ``ocean`` dynamical cores for
MPAS.  Nearly all test cases include calls that launch one of these dynamical
cores.  These runs are configured with namelists and streams files, and one of
the benefits of using COMPASS over attempting to run one of the MPAS components
directly is that COMPASS begins with default values for all namelists and
streams, modifying only those options where the default is not appropriate.
In this way, COMPASS requires little alteration as MPAS itself evolves and new
functionality is added.

.. toctree::
   :caption: User's guide
   :maxdepth: 2

   users_guide/overview

.. toctree::
   :caption: Land Ice
   :maxdepth: 2

   landice/overview

.. toctree::
   :caption: Ocean
   :maxdepth: 2

   ocean/overview

.. toctree::
   :caption: Machines
   :maxdepth: 2

   machines/overview

.. toctree::
   :caption: Developer's guide
   :maxdepth: 2

   developers_guide/overview

.. toctree::
   :caption: Legacy Version
   :maxdepth: 2

    legacy/index