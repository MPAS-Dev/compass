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

   overview
   list_testcases
   setup_testcase
   clean_testcase
   manage_regression_suite

.. toctree::
   :caption: Ocean
   :maxdepth: 2

   ocean
   ocean_testcases/index

.. toctree::
   :caption: Developer's guide
   :maxdepth: 2

   config
   driver_script
   template
   regression_suite
   run_config
   building_docs
