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

   api_examples
   api_ocean


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

   download
   symlink

model
^^^^^

.. currentmodule:: compass.model

.. autosummary::
   :toctree: generated/

   symlink_model
   partition
   run_model

namelist
^^^^^^^^

.. currentmodule:: compass.namelist

.. autosummary::
   :toctree: generated/

   parse_replacements
   generate
   update

parallel
^^^^^^^^

.. currentmodule:: compass.parallel

.. autosummary::
   :toctree: generated/

   get_available_cores_and_nodes
   update_namelist_pio

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

   read
   generate

testcase
^^^^^^^^

.. currentmodule:: compass.testcase

.. autosummary::
   :toctree: generated/

   get_step_default
   get_testcase_default
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
