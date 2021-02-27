ocean
-----

Configurations
^^^^^^^^^^^^^^

.. currentmodule:: compass.ocean.tests

.. autosummary::
   :toctree: generated/

   collect

baroclinic_channel
~~~~~~~~~~~~~~~~~~

.. currentmodule:: compass.ocean.tests.baroclinic_channel

.. autosummary::
   :toctree: generated/

   collect
   configure

   decomp_test.collect
   decomp_test.configure
   decomp_test.run

   default.collect
   default.configure
   default.run

   restart_test.collect
   restart_test.configure
   restart_test.run

   rpe_test.collect
   rpe_test.configure
   rpe_test.run
   rpe_test.analysis.collect
   rpe_test.analysis.setup
   rpe_test.analysis.run

   threads_test.collect
   threads_test.configure
   threads_test.run

   forward.collect
   forward.setup
   forward.run

   initial_state.collect
   initial_state.setup
   initial_state.run

global_ocean
~~~~~~~~~~~~


test cases and steps
''''''''''''''''''''

.. currentmodule:: compass.ocean.tests.global_ocean

.. autosummary::
   :toctree: generated/

   collect
   configure

   analysis_test.collect
   analysis_test.configure
   analysis_test.run

   daily_output_test.collect
   daily_output_test.configure
   daily_output_test.run

   decomp_test.collect
   decomp_test.configure
   decomp_test.run

   files_for_e3sm.collect
   files_for_e3sm.configure
   files_for_e3sm.run
   files_for_e3sm.ocean_graph_partition.collect
   files_for_e3sm.ocean_graph_partition.run
   files_for_e3sm.ocean_initial_condition.collect
   files_for_e3sm.ocean_initial_condition.run
   files_for_e3sm.scrip.collect
   files_for_e3sm.scrip.run
   files_for_e3sm.seaice_initial_condition.collect
   files_for_e3sm.seaice_initial_condition.run
   files_for_e3sm.diagnostics_files.collect
   files_for_e3sm.diagnostics_files.run

   init.collect
   init.configure
   init.run
   init.initial_state.collect
   init.initial_state.setup
   init.initial_state.run
   init.ssh_adjustment.collect
   init.ssh_adjustment.setup
   init.ssh_adjustment.run

   mesh.collect
   mesh.configure
   mesh.run
   mesh.cull.cull_mesh
   mesh.mesh.collect
   mesh.mesh.setup
   mesh.mesh.run

   mesh.ec30to60.build_cell_width_lat_lon
   mesh.ec30to60.spinup.collect
   mesh.ec30to60.spinup.configure
   mesh.ec30to60.spinup.run

   mesh.qu240.build_cell_width_lat_lon
   mesh.qu240.spinup.collect
   mesh.qu240.spinup.configure
   mesh.qu240.spinup.run

   performance_test.collect
   performance_test.configure
   performance_test.run

   restart_test.collect
   restart_test.configure
   restart_test.run

   threads_test.collect
   threads_test.configure
   threads_test.run

   forward.collect
   forward.setup
   forward.run

global_ocean framework
''''''''''''''''''''''

.. currentmodule:: compass.ocean.tests.global_ocean

.. autosummary::
   :toctree: generated/

   description.get_description

   metadata.get_e3sm_mesh_names
   metadata.add_mesh_and_init_metadata

   subdir.get_init_sudbdir
   subdir.get_forward_sudbdir
   subdir.get_mesh_relative_path
   subdir.get_initial_condition_relative_path

ice_shelf_2d
~~~~~~~~~~~~

.. currentmodule:: compass.ocean.tests.ice_shelf_2d

.. autosummary::
   :toctree: generated/

   collect
   configure

   default.collect
   default.configure
   default.run

   restart_test.collect
   restart_test.configure
   restart_test.run

   forward.collect
   forward.setup
   forward.run

   initial_state.collect
   initial_state.run

   ssh_adjustment.collect
   ssh_adjustment.setup
   ssh_adjustment.run

ziso
~~~~

.. currentmodule:: compass.ocean.tests.ziso

.. autosummary::
   :toctree: generated/

   collect
   configure

   default.collect
   default.configure
   default.run

   with_frazil.collect
   with_frazil.configure
   with_frazil.run

   forward.collect
   forward.setup
   forward.run

   initial_state.collect
   initial_state.run

ocean framework
^^^^^^^^^^^^^^^

.. currentmodule:: compass.ocean

.. autosummary::
   :toctree: generated/

   iceshelf.compute_land_ice_pressure_and_draft
   iceshelf.adjust_ssh

   particles.write
   particles.remap_particles

   plot.plot_initial_state
   plot.plot_vertical_grid

   vertical.generate_grid
   vertical.write_grid
   vertical.zstar.compute_layer_thickness_and_zmid
