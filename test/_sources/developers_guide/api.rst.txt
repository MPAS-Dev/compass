.. _dev_api:

#############
API reference
#############

This page provides an auto-generated summary of the ``compass`` API. For more
details and examples, refer to the relevant sections in the main part of the
documentation.

Cores
=====

.. toctree::
   :titlesonly:
   :maxdepth: 1

   examples/api
   landice/api
   ocean/api


compass framework
=================

Command-line interface
^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: compass

.. autosummary::
   :toctree: generated/

   __main__.main


list
~~~~

.. currentmodule:: compass.list

.. autosummary::
   :toctree: generated/

   list_cases

setup
~~~~~

.. currentmodule:: compass.setup

.. autosummary::
   :toctree: generated/

   setup_cases
   setup_case

clean
~~~~~

.. currentmodule:: compass.clean

.. autosummary::
   :toctree: generated/

   clean_cases

suite
~~~~~

.. currentmodule:: compass.suite

.. autosummary::
   :toctree: generated/

   setup_suite
   clean_suite
   run_suite

config
^^^^^^

.. currentmodule:: compass.config

.. autosummary::
   :toctree: generated/

   duplicate_config
   add_config
   ensure_absolute_paths
   get_source_file

io
^^

.. currentmodule:: compass.io

.. autosummary::
   :toctree: generated/

   add_input_file
   add_output_file
   download
   symlink
   process_step_inputs_and_outputs

model
^^^^^

.. currentmodule:: compass.model

.. autosummary::
   :toctree: generated/

   add_model_as_input
   run_model
   partition
   update_namelist_pio
   make_graph_file

namelist
^^^^^^^^

.. currentmodule:: compass.namelist

.. autosummary::
   :toctree: generated/

   add_namelist_file
   add_namelist_options
   generate_namelist
   update

parallel
^^^^^^^^

.. currentmodule:: compass.parallel

.. autosummary::
   :toctree: generated/

   get_available_cores_and_nodes

provenance
^^^^^^^^^^

.. currentmodule:: compass.provenance

.. autosummary::
   :toctree: generated/

   write

streams
^^^^^^^

.. currentmodule:: compass.streams

.. autosummary::
   :toctree: generated/

   add_streams_file
   generate_streams

testcase
^^^^^^^^

.. currentmodule:: compass.testcase

.. autosummary::
   :toctree: generated/

   add_testcase
   set_testcase_subdir
   add_step
   run_steps
   run_step
   generate_run

testcases
^^^^^^^^^

.. currentmodule:: compass.testcases

.. autosummary::
   :toctree: generated/

   collect
   validate

validate
^^^^^^^^

.. currentmodule:: compass.validate

.. autosummary::
   :toctree: generated/

   compare_variables
   compare_timers
