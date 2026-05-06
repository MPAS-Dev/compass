.. _dev_api:

#############
API reference
#############

This page provides an auto-generated summary of the ``compass`` API. For more
details and examples, refer to the relevant sections in the main part of the
documentation.

MPAS Cores
==========

.. toctree::
   :titlesonly:
   :maxdepth: 1

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
   list_machines
   list_suites

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

run
~~~

.. currentmodule:: compass.run.serial

.. autosummary::
   :toctree: generated/

   run_tests
   run_single_step


cache
~~~~~

.. currentmodule:: compass.cache

.. autosummary::
   :toctree: generated/

   update_cache


Base Classes
^^^^^^^^^^^^

mpas_core
~~~~~~~~~

.. currentmodule:: compass

.. autosummary::
   :toctree: generated/

   MpasCore
   MpasCore.add_test_group

testgroup
~~~~~~~~~

.. currentmodule:: compass

.. autosummary::
   :toctree: generated/

   TestGroup
   TestGroup.add_test_case

testcase
^^^^^^^^

.. currentmodule:: compass

.. autosummary::
   :toctree: generated/

   TestCase
   TestCase.configure
   TestCase.run
   TestCase.validate
   TestCase.add_step

step
^^^^

.. currentmodule:: compass

.. autosummary::
   :toctree: generated/

   Step
   Step.set_resources
   Step.constrain_resources
   Step.setup
   Step.runtime_setup
   Step.run
   Step.add_input_file
   Step.add_output_file
   Step.add_model_as_input
   Step.add_namelist_file
   Step.add_namelist_options
   Step.update_namelist_at_runtime
   Step.update_namelist_pio
   Step.add_streams_file
   Step.update_streams_at_runtime

config
^^^^^^

.. currentmodule:: compass.config

.. autosummary::
   :toctree: generated/

   CompassConfigParser

io
^^

.. currentmodule:: compass.io

.. autosummary::
   :toctree: generated/

   download
   symlink
   package_path

logging
^^^^^^^

.. currentmodule:: compass.logging

.. autosummary::
   :toctree: generated/

   log_method_call

mesh
^^^^

.. currentmodule:: compass.mesh

.. autosummary::
   :toctree: generated/

   spherical.SphericalBaseStep
   spherical.SphericalBaseStep.setup
   spherical.SphericalBaseStep.run
   spherical.SphericalBaseStep.save_and_plot_cell_width

   QuasiUniformSphericalMeshStep
   QuasiUniformSphericalMeshStep.setup
   QuasiUniformSphericalMeshStep.run
   QuasiUniformSphericalMeshStep.build_cell_width_lat_lon
   QuasiUniformSphericalMeshStep.make_jigsaw_mesh

   IcosahedralMeshStep
   IcosahedralMeshStep.setup
   IcosahedralMeshStep.run
   IcosahedralMeshStep.make_jigsaw_mesh
   IcosahedralMeshStep.build_subdivisions_cell_width_lat_lon
   IcosahedralMeshStep.get_subdivisions
   IcosahedralMeshStep.get_cell_width

model
^^^^^

.. currentmodule:: compass.model

.. autosummary::
   :toctree: generated/

   run_model
   partition
   make_graph_file

mpas_cores
^^^^^^^^^^

.. currentmodule:: compass.mpas_cores

.. autosummary::
   :toctree: generated/

   get_mpas_cores

parallel
^^^^^^^^

.. currentmodule:: compass.parallel

.. autosummary::
   :toctree: generated/

   get_available_parallel_resources
   set_cores_per_node
   run_command

provenance
^^^^^^^^^^

.. currentmodule:: compass.provenance

.. autosummary::
   :toctree: generated/

   write

validate
^^^^^^^^

.. currentmodule:: compass.validate

.. autosummary::
   :toctree: generated/

   compare_variables
   compare_timers
