COMPASS
=======

Configuration Of Model for Prediction Across Scales Setups (COMPASS) is an
automated system to set up test cases that match the MPAS-Model repository. All
namelists and streams files begin with the default generated from the
Registry.xml file, and only the changes relevant to the particular test case are
altered in those files.

.. toctree::
   :caption: User's guide
   :maxdepth: 2

   users_guide/overview
   users_guide/scripts

.. toctree::
   :caption: Ocean
   :maxdepth: 2

   ocean/introduction
   ocean/test_cases/index

.. toctree::
   :caption: Machine-specific instructions
   :maxdepth: 2

   machine_specific_instructions/slurm
   machine_specific_instructions/lanl
   machine_specific_instructions/nersc
   machine_specific_instructions/anvil
   machine_specific_instructions/linux
   machine_specific_instructions/osx

.. toctree::
   :caption: Developer's guide
   :maxdepth: 2

   developers_guide/config
   developers_guide/driver_script
   developers_guide/template
   developers_guide/regression_suite
   developers_guide/run_config
   developers_guide/building_docs
