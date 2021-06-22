ocean
-----

.. currentmodule:: compass.ocean

.. autosummary::
   :toctree: generated/

   Ocean

Test Groups
^^^^^^^^^^^

baroclinic_channel
~~~~~~~~~~~~~~~~~~

.. currentmodule:: compass.ocean.tests.baroclinic_channel

.. autosummary::
   :toctree: generated/

   BaroclinicChannel
   configure

   decomp_test.DecompTest
   decomp_test.DecompTest.configure
   decomp_test.DecompTest.run

   default.Default
   default.Default.configure
   default.Default.run

   restart_test.RestartTest
   restart_test.RestartTest.configure
   restart_test.RestartTest.run

   rpe_test.RpeTest
   rpe_test.RpeTest.configure
   rpe_test.RpeTest.run
   rpe_test.analysis.Analysis
   rpe_test.analysis.Analysis.setup
   rpe_test.analysis.Analysis.run

   threads_test.ThreadsTest
   threads_test.ThreadsTest.configure
   threads_test.ThreadsTest.run

   forward.Forward
   forward.Forward.setup
   forward.Forward.run

   initial_state.InitialState
   initial_state.InitialState.setup
   initial_state.InitialState.run


global_convergence
~~~~~~~~~~~~~~~~~~

.. currentmodule:: compass.ocean.tests.global_convergence

.. autosummary::
   :toctree: generated/

   GlobalConvergence

cosine_bell
'''''''''''

.. currentmodule:: compass.ocean.tests.global_convergence.cosine_bell

.. autosummary::
   :toctree: generated/

   CosineBell
   CosineBell.configure
   CosineBell.run

   mesh.Mesh
   mesh.Mesh.run
   mesh.Mesh.build_cell_width_lat_lon

   init.Init
   init.Init.run

   forward.Forward
   forward.Forward.setup
   forward.Forward.run
   forward.Forward.get_dt

   analysis.Analysis
   analysis.Analysis.run
   analysis.Analysis.rmse


global_ocean
~~~~~~~~~~~~


test cases and steps
''''''''''''''''''''

.. currentmodule:: compass.ocean.tests.global_ocean

.. autosummary::
   :toctree: generated/

   GlobalOcean
   configure

   analysis_test.AnalysisTest
   analysis_test.AnalysisTest.configure
   analysis_test.AnalysisTest.run

   daily_output_test.DailyOutputTest
   daily_output_test.DailyOutputTest.configure
   daily_output_test.DailyOutputTest.run

   decomp_test.DecompTest
   decomp_test.DecompTest.configure
   decomp_test.DecompTest.run

   files_for_e3sm.FilesForE3SM
   files_for_e3sm.FilesForE3SM.configure
   files_for_e3sm.FilesForE3SM.run
   files_for_e3sm.ocean_graph_partition.OceanGraphPartition
   files_for_e3sm.ocean_graph_partition.OceanGraphPartition.run
   files_for_e3sm.ocean_initial_condition.OceanInitialCondition
   files_for_e3sm.ocean_initial_condition.OceanInitialCondition.run
   files_for_e3sm.scrip.Scrip
   files_for_e3sm.scrip.Scrip.run
   files_for_e3sm.seaice_initial_condition.SeaiceInitialCondition
   files_for_e3sm.seaice_initial_condition.SeaiceInitialCondition.run
   files_for_e3sm.diagnostics_files.DiagnosticsFiles
   files_for_e3sm.diagnostics_files.DiagnosticsFiles.run

   init.Init
   init.Init.configure
   init.Init.run
   init.initial_state.InitialState
   init.initial_state.InitialState.setup
   init.initial_state.InitialState.run
   init.ssh_adjustment.SshAdjustment
   init.ssh_adjustment.SshAdjustment.setup
   init.ssh_adjustment.SshAdjustment.run

   mesh.Mesh
   mesh.Mesh.configure
   mesh.Mesh.run
   mesh.cull.cull_mesh
   mesh.mesh.MeshStep
   mesh.mesh.MeshStep.setup
   mesh.mesh.MeshStep.run
   mesh.mesh.MeshStep.build_cell_width_lat_lon

   mesh.ec30to60.EC30to60Mesh
   mesh.ec30to60.EC30to60Mesh.build_cell_width_lat_lon
   mesh.ec30to60.dynamic_adjustment.EC30to60DynamicAdjustment
   mesh.ec30to60.dynamic_adjustment.EC30to60DynamicAdjustment.configure
   mesh.ec30to60.dynamic_adjustment.EC30to60DynamicAdjustment.run

   mesh.qu240.QU240Mesh
   mesh.qu240.QU240Mesh.build_cell_width_lat_lon
   mesh.qu240.dynamic_adjustment.QU240DynamicAdjustment
   mesh.qu240.dynamic_adjustment.QU240DynamicAdjustment.configure
   mesh.qu240.dynamic_adjustment.QU240DynamicAdjustment.run

   mesh.so12to60.SO12to60Mesh
   mesh.so12to60.SO12to60Mesh.build_cell_width_lat_lon
   mesh.so12to60.dynamic_adjustment.SO12to60DynamicAdjustment
   mesh.so12to60.dynamic_adjustment.SO12to60DynamicAdjustment.configure
   mesh.so12to60.dynamic_adjustment.SO12to60DynamicAdjustment.run

   mesh.wc14.WC14Mesh
   mesh.wc14.WC14Mesh.build_cell_width_lat_lon
   mesh.wc14.dynamic_adjustment.WC14DynamicAdjustment
   mesh.wc14.dynamic_adjustment.WC14DynamicAdjustment.configure
   mesh.wc14.dynamic_adjustment.WC14DynamicAdjustment.run

   performance_test.PerformanceTest
   performance_test.PerformanceTest.configure
   performance_test.PerformanceTest.run

   restart_test.RestartTest
   restart_test.RestartTest.configure
   restart_test.RestartTest.run

   threads_test.ThreadsTest
   threads_test.ThreadsTest.configure
   threads_test.ThreadsTest.run

   dynamic_adjustment.DynamicAdjustment
   dynamic_adjustment.DynamicAdjustment.run

   forward.ForwardTestCase
   forward.ForwardTestCase.configure
   forward.ForwardTestCase.run

   forward.ForwardStep
   forward.ForwardStep.setup
   forward.ForwardStep.run


global_ocean framework
''''''''''''''''''''''

.. currentmodule:: compass.ocean.tests.global_ocean

.. autosummary::
   :toctree: generated/

   configure.configure_global_ocean
   metadata.get_e3sm_mesh_names
   metadata.add_mesh_and_init_metadata


gotm
~~~~

.. currentmodule:: compass.ocean.tests.gotm

.. autosummary::
   :toctree: generated/

   Gotm

default
'''''''

.. currentmodule:: compass.ocean.tests.gotm.default

.. autosummary::
   :toctree: generated/

   Default
   Default.validate

   init.Init
   init.Init.run

   forward.Forward
   forward.Forward.run

   analysis.Analysis
   analysis.Analysis.run


ice_shelf_2d
~~~~~~~~~~~~

.. currentmodule:: compass.ocean.tests.ice_shelf_2d

.. autosummary::
   :toctree: generated/

   IceShelf2d
   configure

   default.Default
   default.Default.configure
   default.Default.run

   restart_test.RestartTest
   restart_test.RestartTest.configure
   restart_test.RestartTest.run

   forward.Forward
   forward.Forward.setup
   forward.Forward.run

   initial_state.InitialState
   initial_state.InitialState.run

   ssh_adjustment.SshAdjustment
   ssh_adjustment.SshAdjustment.setup
   ssh_adjustment.SshAdjustment.run

ziso
~~~~

.. currentmodule:: compass.ocean.tests.ziso

.. autosummary::
   :toctree: generated/

   Ziso
   configure

   default.Default
   default.Default.configure
   default.Default.run

   with_frazil.WithFrazil
   with_frazil.WithFrazil.configure
   with_frazil.WithFrazil.run

   forward.Forward
   forward.Forward.setup
   forward.Forward.run

   initial_state.InitialState
   initial_state.InitialState.run

ocean framework
^^^^^^^^^^^^^^^

.. currentmodule:: compass.ocean

.. autosummary::
   :toctree: generated/

   haney.compute_haney_number

   iceshelf.compute_land_ice_pressure_and_draft
   iceshelf.adjust_ssh

   particles.write
   particles.remap_particles

   plot.plot_initial_state
   plot.plot_vertical_grid

   vertical.init_vertical_coord
   vertical.grid_1d.generate_1d_grid
   vertical.grid_1d.write_1d_grid
   vertical.partial_cells.alter_bottom_depth
   vertical.partial_cells.alter_ssh
   vertical.zlevel.init_z_level_vertical_coord
   vertical.zlevel.compute_min_max_level_cell
   vertical.zlevel.compute_z_level_layer_thickness
   vertical.zlevel.compute_z_level_resting_thickness
   vertical.zstar.init_z_star_vertical_coord
