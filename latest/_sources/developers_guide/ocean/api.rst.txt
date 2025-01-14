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

buttermilk_bay
~~~~~~~~~~~~~~

.. currentmodule:: compass.ocean.tests.buttermilk_bay

.. autosummary::
   :toctree: generated/

   ButtermilkBay

   default.Default
   default.Default.configure
   default.Default.update_cores
   default.Default.validate

   forward.Forward
   forward.Forward.run
   forward.Forward.setup
   forward.Forward.get_dt

   initial_state.InitialState
   initial_state.InitialState.run

   viz.Viz
   viz.Viz.run
   viz.Viz.get_points
   viz.Viz.timeseries_plots
   viz.Viz.contour_plots


baroclinic_gyre
~~~~~~~~~~~~~~~

.. currentmodule:: compass.ocean.tests.baroclinic_gyre

.. autosummary::
   :toctree: generated/

   BaroclinicGyre

   GyreTestCase
   GyreTestCase.configure

   cull_mesh.CullMesh
   cull_mesh.CullMesh.run

   forward.Forward
   forward.Forward.run

   initial_state.InitialState
   initial_state.InitialState.run

   moc.Moc
   moc.Moc.run

   viz.Viz
   viz.Viz.run

dam_break
~~~~~~~~~

.. currentmodule:: compass.ocean.tests.dam_break

.. autosummary::
   :toctree: generated/

   DamBreak

   default.Default
   default.Default.configure

   forward.Forward
   forward.Forward.run

   initial_state.InitialState
   initial_state.InitialState.run

   viz.Viz
   viz.Viz.run


drying_slope
~~~~~~~~~~~~

.. currentmodule:: compass.ocean.tests.drying_slope

.. autosummary::
   :toctree: generated/

   DryingSlope

   analysis.Analysis
   analysis.Analysis.run

   convergence.Convergence
   convergence.Convergence.validate

   decomp.Decomp
   decomp.Decomp.validate

   default.Default
   default.Default.validate

   forward.Forward
   forward.Forward.run

   initial_state.InitialState
   initial_state.InitialState.run

   loglaw.LogLaw
   loglaw.LogLaw.configure
   loglaw.LogLaw.validate

   viz.Viz
   viz.Viz.run

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
   files_for_e3sm.ocean_mesh.OceanMesh
   files_for_e3sm.ocean_mesh.OceanMesh.run
   files_for_e3sm.scrip.Scrip
   files_for_e3sm.scrip.Scrip.run
   files_for_e3sm.seaice_graph_partition.SeaiceGraphPartition
   files_for_e3sm.seaice_graph_partition.SeaiceGraphPartition.run
   files_for_e3sm.seaice_initial_condition.SeaiceInitialCondition
   files_for_e3sm.seaice_initial_condition.SeaiceInitialCondition.run
   files_for_e3sm.seaice_mesh.SeaiceMesh
   files_for_e3sm.seaice_mesh.SeaiceMesh.run
   files_for_e3sm.e3sm_to_cmip_maps.E3smToCmipMaps
   files_for_e3sm.e3sm_to_cmip_maps.E3smToCmipMaps.run
   files_for_e3sm.diagnostic_maps.DiagnosticMaps
   files_for_e3sm.diagnostic_maps.DiagnosticMaps.run
   files_for_e3sm.diagnostic_masks.DiagnosticMasks
   files_for_e3sm.diagnostic_masks.DiagnosticMasks.run
   files_for_e3sm.remap_ice_shelf_melt.RemapIceShelfMelt
   files_for_e3sm.remap_ice_shelf_melt.RemapIceShelfMelt.run
   files_for_e3sm.remap_sea_surface_salinity_restoring.RemapSeaSurfaceSalinityRestoring
   files_for_e3sm.remap_sea_surface_salinity_restoring.RemapSeaSurfaceSalinityRestoring.run
   files_for_e3sm.remap_iceberg_climatology.RemapIcebergClimatology
   files_for_e3sm.remap_iceberg_climatology.RemapIcebergClimatology.run
   files_for_e3sm.remap_tidal_mixing.RemapTidalMixing
   files_for_e3sm.remap_tidal_mixing.RemapTidalMixing.run
   files_for_e3sm.write_coeffs_reconstruct.WriteCoeffsReconstruct
   files_for_e3sm.write_coeffs_reconstruct.WriteCoeffsReconstruct.run

   init.Init
   init.Init.configure
   init.Init.run
   init.initial_state.InitialState
   init.initial_state.InitialState.setup
   init.initial_state.InitialState.run
   init.remap_ice_shelf_melt.RemapIceShelfMelt
   init.remap_ice_shelf_melt.RemapIceShelfMelt.run
   init.remap_ice_shelf_melt.remap_paolo
   init.remap_ice_shelf_melt.remap_adusumilli
   init.ssh_adjustment.SshAdjustment
   init.ssh_adjustment.SshAdjustment.setup
   init.ssh_adjustment.SshAdjustment.run
   init.ssh_from_surface_density.SshFromSurfaceDensity
   init.ssh_from_surface_density.SshFromSurfaceDensity.run

   mesh.Mesh
   mesh.Mesh.configure
   mesh.Mesh.run

   mesh.ec30to60.EC30to60BaseMesh
   mesh.ec30to60.EC30to60BaseMesh.build_cell_width_lat_lon

   mesh.qu.IcosMeshFromConfigStep
   mesh.qu.IcosMeshFromConfigStep.setup
   mesh.qu.QUMeshFromConfigStep
   mesh.qu.QUMeshFromConfigStep.setup

   mesh.rrs6to18.RRS6to18BaseMesh
   mesh.rrs6to18.RRS6to18BaseMesh.build_cell_width_lat_lon

   mesh.so12to30.SO12to30BaseMesh
   mesh.so12to30.SO12to30BaseMesh.build_cell_width_lat_lon

   mesh.wc14.WC14BaseMesh
   mesh.wc14.WC14BaseMesh.build_cell_width_lat_lon

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
   dynamic_adjustment.DynamicAdjustment.validate

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

   metadata.get_e3sm_mesh_names
   metadata.add_mesh_and_init_metadata

   tasks.get_ntasks_from_cell_count


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

hurricane
~~~~~~~~~

test cases and steps
''''''''''''''''''''

.. currentmodule:: compass.ocean.tests.hurricane

.. autosummary::
   :toctree: generated/

   Hurricane
   configure

   mesh.Mesh
   mesh.Mesh.configure
   mesh.Mesh.run

   mesh.dequ120at30cr10rr2.DEQU120at30cr10rr2BaseMesh
   mesh.dequ120at30cr10rr2.DEQU120at30cr10rr2BaseMesh.build_cell_width_lat_lon

   init.Init
   init.Init.configure
   init.Init.run
   init.create_pointstats_file.CreatePointstatsFile
   init.create_pointstats_file.CreatePointstatsFile.create_pointstats_file
   init.create_pointstats_file.CreatePointstatsFile.run
   init.initial_state.InitialState
   init.initial_state.InitialState.setup
   init.initial_state.InitialState.run
   init.interpolate_atm_forcing.InterpolateAtmForcing
   init.interpolate_atm_forcing.InterpolateAtmForcing.interpolate_data_to_grid
   init.interpolate_atm_forcing.InterpolateAtmForcing.plot_interp_data
   init.interpolate_atm_forcing.InterpolateAtmForcing.write_to_file
   init.interpolate_atm_forcing.InterpolateAtmForcing.run

   forward.Forward
   forward.Forward.configure
   forward.Forward.run
   forward.forward.ForwardStep
   forward.forward.ForwardStep.setup
   forward.forward.ForwardStep.run

   analysis.Analysis
   analysis.Analysis.setup
   analysis.Analysis.read_pointstats
   analysis.Analysis.read_station_data
   analysis.Analysis.read_station_file
   analysis.Analysis.run

   lts.mesh.lts_regions.LTSRegionsStep
   lts.mesh.lts_regions.LTSRegionsStep.setup
   lts.mesh.lts_regions.LTSRegionsStep.run
   lts.mesh.lts_regions.label_mesh

   lts.init.topographic_wave_drag.ComputeTopographicWaveDrag
   lts.init.topographic_wave_drag.ComputeTopographicWaveDrag.interpolate_data_to_grid
   lts.init.topographic_wave_drag.ComputeTopographicWaveDrag.write_to_file
   lts.init.topographic_wave_drag.ComputeTopographicWaveDrag.run

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

   viz.Viz
   viz.Viz.run

internal_wave
~~~~~~~~~~~~~

.. currentmodule:: compass.ocean.tests.internal_wave

.. autosummary::
   :toctree: generated/

   InternalWave

   default.Default
   default.Default.validate

   rpe_test.RpeTest

   rpe_test.analysis.Analysis
   rpe_test.analysis.Analysis.run

   ten_day_test.TenDayTest
   ten_day_test.TenDayTest.validate

   forward.Forward
   forward.Forward.run

   initial_state.InitialState
   initial_state.InitialState.run

   viz.Viz
   viz.Viz.run


isomip_plus
~~~~~~~~~~~

.. currentmodule:: compass.ocean.tests.isomip_plus

.. autosummary::
   :toctree: generated/

   IsomipPlus

   isomip_plus_test.IsomipPlusTest
   isomip_plus_test.IsomipPlusTest.configure
   isomip_plus_test.IsomipPlusTest.run

   viz.Viz
   viz.Viz.run
   viz.file_complete

   viz.plot.TimeSeriesPlotter
   viz.plot.TimeSeriesPlotter.plot_melt_time_series
   viz.plot.TimeSeriesPlotter.plot_time_series

   viz.plot.MoviePlotter
   viz.plot.MoviePlotter.plot_barotropic_streamfunction
   viz.plot.MoviePlotter.plot_overturning_streamfunction
   viz.plot.MoviePlotter.plot_melt_rates
   viz.plot.MoviePlotter.plot_ice_shelf_boundary_variables
   viz.plot.MoviePlotter.plot_temperature
   viz.plot.MoviePlotter.plot_salinity
   viz.plot.MoviePlotter.plot_potential_density
   viz.plot.MoviePlotter.plot_horiz_series
   viz.plot.MoviePlotter.plot_3d_field_top_bot_section
   viz.plot.MoviePlotter.plot_layer_interfaces
   viz.plot.MoviePlotter.images_to_movies

   evap.update_evaporation_flux

   forward.Forward
   forward.Forward.setup
   forward.Forward.run

   geom.interpolate_ocean_mask
   geom.interpolate_geom

   initial_state.InitialState
   initial_state.InitialState.run

   misomip.Misomip
   misomip.Misomip.run

   process_geom.ProcessGeom
   process_geom.ProcessGeom.run

   ssh_adjustment.SshAdjustment
   ssh_adjustment.SshAdjustment.setup
   ssh_adjustment.SshAdjustment.run

   streamfunction.Streamfunction
   streamfunction.Streamfunction.run

lock_exchange
~~~~~~~~~~~~~

.. currentmodule:: compass.ocean.tests.lock_exchange

.. autosummary::
   :toctree: generated/

   LockExchange

   hydro.Hydro

   nonhydro.Nonhydro

   forward.Forward
   forward.Forward.run

   initial_state.InitialState
   initial_state.InitialState.run

   visualize.Visualize
   visualize.Visualize.run

merry_go_round
~~~~~~~~~~~~~~

.. currentmodule:: compass.ocean.tests.merry_go_round

.. autosummary::
   :toctree: generated/

   MerryGoRound

   default.Default
   default.Default.validate

   convergence_test.analysis.Analysis
   convergence_test.analysis.Analysis.run

   forward.Forward
   forward.Forward.run

   initial_state.InitialState
   initial_state.InitialState.run

   viz.Viz
   viz.Viz.run

nonhydro
~~~~~~~~

.. currentmodule:: compass.ocean.tests.nonhydro

.. autosummary::
   :toctree: generated/

   Nonhydro

   stratified_seiche.StratifiedSeiche
   stratified_seiche.StratifiedSeiche.configure
   stratified_seiche.initial_state.InitialState
   stratified_seiche.initial_state.InitialState.setup
   stratified_seiche.initial_state.InitialState.run
   stratified_seiche.forward.Forward
   stratified_seiche.forward.Forward.setup
   stratified_seiche.forward.Forward.run
   stratified_seiche.visualize.Visualize
   stratified_seiche.visualize.Visualize.setup
   stratified_seiche.visualize.Visualize.run

   solitary_wave.SolitaryWave
   solitary_wave.SolitaryWave.configure
   solitary_wave.initial_state.InitialState
   solitary_wave.initial_state.InitialState.setup
   solitary_wave.initial_state.InitialState.run
   solitary_wave.forward.Forward
   solitary_wave.forward.Forward.setup
   solitary_wave.forward.Forward.run
   solitary_wave.visualize.Visualize
   solitary_wave.visualize.Visualize.setup
   solitary_wave.visualize.Visualize.run

overflow
~~~~~~~~

.. currentmodule:: compass.ocean.tests.overflow

.. autosummary::
   :toctree: generated/

   Overflow

   default.Default

   rpe_test.RpeTest

   rpe_test.analysis.Analysis
   rpe_test.analysis.Analysis.run

   forward.Forward
   forward.Forward.run

   initial_state_from_init_mode.InitialStateFromInitMode
   initial_state_from_init_mode.InitialStateFromInitMode.run

   initial_state.InitialState
   initial_state.InitialState.run

   nonhydro.Nonhydro
   nonhydro.forward.Forward
   nonhydro.forward.Forward.run

   hydro_vs_nonhydro.HydroVsNonhydro
   hydro_vs_nonhydro.forward.Forward
   hydro_vs_nonhydro.forward.Forward.run
   hydro_vs_nonhydro.visualize.Visualize
   hydro_vs_nonhydro.visualize.Visualize.run

parabolic_bowl
~~~~~~~~~~~~~~

.. currentmodule:: compass.ocean.tests.parabolic_bowl

.. autosummary::
   :toctree: generated/

   ParabolicBowl

   default.Default
   default.Default.configure
   default.Default.update_cores
   default.Default.validate

   forward.Forward
   forward.Forward.run
   forward.Forward.setup
   forward.Forward.get_dt

   initial_state.InitialState
   initial_state.InitialState.run

   viz.Viz
   viz.Viz.run
   viz.Viz.get_points
   viz.Viz.timeseries_plots
   viz.Viz.inject_exact_solution
   viz.Viz.contour_plots
   viz.Viz.rmse_plots
   viz.Viz.compute_rmse
   viz.Viz.exact_solution

planar_convergence
~~~~~~~~~~~~~~~~~~

.. currentmodule:: compass.ocean.tests.planar_convergence

.. autosummary::
   :toctree: generated/

   PlanarConvergence

   conv_init.ConvInit
   conv_init.ConvInit.run

   conv_test_case.ConvTestCase
   conv_test_case.ConvTestCase.configure
   conv_test_case.ConvTestCase.run
   conv_test_case.ConvTestCase.update_cores

   forward.Forward
   forward.Forward.setup
   forward.Forward.run
   forward.Forward.get_dt_duration

horizontal_advection
''''''''''''''''''''

.. currentmodule:: compass.ocean.tests.planar_convergence.horizontal_advection

.. autosummary::
   :toctree: generated/

   HorizontalAdvection
   HorizontalAdvection.configure
   HorizontalAdvection.run

   init.Init
   init.Init.run

   analysis.Analysis
   analysis.Analysis.run
   analysis.Analysis.rmse

soma
~~~~

.. currentmodule:: compass.ocean.tests.soma
.. autosummary::
   :toctree: generated/

   Soma

   soma_test_case.SomaTestCase
   soma_test_case.SomaTestCase.validate

   analysis.Analysis
   analysis.Analysis.run

   forward.Forward
   forward.Forward.run

   initial_state.InitialState
   initial_state.InitialState.run

sphere_transport
~~~~~~~~~~~~~~~~

.. currentmodule:: compass.ocean.tests.sphere_transport

.. autosummary::
   :toctree: generated/

   SphereTransport


correlated_tracers_2d
'''''''''''''''''''''

.. currentmodule:: compass.ocean.tests.sphere_transport.correlated_tracers_2d

.. autosummary::
   :toctree: generated/

   CorrelatedTracers2D
   CorrelatedTracers2D.configure
   CorrelatedTracers2D.run

   mesh.Mesh
   mesh.Mesh.run
   mesh.Mesh.build_cell_width_lat_lon

   init.Init
   init.Init.run

   forward.Forward
   forward.Forward.setup
   forward.Forward.run
   forward.Forward.get_timestep_str

   analysis.Analysis
   analysis.Analysis.run

divergent_2d
''''''''''''

.. currentmodule:: compass.ocean.tests.sphere_transport.divergent_2d

.. autosummary::
   :toctree: generated/

   Divergent2D
   Divergent2D.configure
   Divergent2D.run

   mesh.Mesh
   mesh.Mesh.run
   mesh.Mesh.build_cell_width_lat_lon

   init.Init
   init.Init.run

   forward.Forward
   forward.Forward.setup
   forward.Forward.run
   forward.Forward.get_timestep_str

   analysis.Analysis
   analysis.Analysis.run

nondivergent_2d
'''''''''''''''

.. currentmodule:: compass.ocean.tests.sphere_transport.nondivergent_2d

.. autosummary::
   :toctree: generated/

   Nondivergent2D
   Nondivergent2D.configure
   Nondivergent2D.run

   mesh.Mesh
   mesh.Mesh.run
   mesh.Mesh.build_cell_width_lat_lon

   init.Init
   init.Init.run

   forward.Forward
   forward.Forward.setup
   forward.Forward.run
   forward.Forward.get_timestep_str

   analysis.Analysis
   analysis.Analysis.run

rotation_2d
'''''''''''

.. currentmodule:: compass.ocean.tests.sphere_transport.rotation_2d

.. autosummary::
   :toctree: generated/

   Rotation2D
   Rotation2D.configure
   Rotation2D.run

   mesh.Mesh
   mesh.Mesh.run
   mesh.Mesh.build_cell_width_lat_lon

   init.Init
   init.Init.run

   forward.Forward
   forward.Forward.setup
   forward.Forward.run
   forward.Forward.get_timestep_str

   analysis.Analysis
   analysis.Analysis.run

sphere_transport framework
''''''''''''''''''''''''''

.. currentmodule:: compass.ocean.tests.sphere_transport

.. autosummary::
   :toctree: generated/

   process_output.compute_error_from_output_ncfile
   process_output.compute_convergence_rates
   process_output.print_error_conv_table
   process_output.read_ncl_rgb_file
   process_output.plot_sol
   process_output.make_convergence_arrays
   process_output.print_data_as_csv
   process_output.plot_convergence
   process_output.plot_filament
   process_output.plot_over_and_undershoot_errors

tides
~~~~~

test cases and steps
''''''''''''''''''''

.. currentmodule:: compass.ocean.tests.tides

.. autosummary::
   :toctree: generated/

   Tides
   configure

   mesh.Mesh
   mesh.Mesh.configure
   mesh.Mesh.run

   init.Init
   init.Init.configure
   init.Init.run
   init.remap_bathymetry.RemapBathymetry
   init.remap_bathymetry.RemapBathymetry.run
   init.initial_state.InitialState
   init.initial_state.InitialState.setup
   init.initial_state.InitialState.run
   init.interpolate_wave_drag.InterpolateWaveDrag
   init.interpolate_wave_drag.InterpolateWaveDrag.interpolate_data_to_grid
   init.interpolate_wave_drag.InterpolateWaveDrag.plot_interp_data
   init.interpolate_wave_drag.InterpolateWaveDrag.write_to_file
   init.interpolate_wave_drag.InterpolateWaveDrag.run


   forward.Forward
   forward.Forward.configure
   forward.Forward.run
   forward.forward.ForwardStep
   forward.forward.ForwardStep.setup
   forward.forward.ForwardStep.run

   analysis.Analysis
   analysis.Analysis.setup
   analysis.Analysis.write_coordinate_file
   analysis.Analysis.setup_otps2
   analysis.Analysis.run_otps2
   analysis.Analysis.read_otps2_output
   analysis.Analysis.append_tpxo_data
   analysis.Analysis.check_tpxo_data
   analysis.Analysis.plot
   analysis.Analysis.run

utility
~~~~~~~

.. currentmodule:: compass.ocean.tests.utility

.. autosummary::
   :toctree: generated/

   Utility

   combine_topo.CombineTopo
   combine_topo.Combine
   combine_topo.Combine.setup
   combine_topo.Combine.run

   create_salin_restoring.CreateSalinRestoring
   create_salin_restoring.Combine
   create_salin_restoring.Extrap
   create_salin_restoring.Remap

   cull_restarts.CullRestarts
   cull_restarts.Cull
   cull_restarts.Cull.run

   extrap_woa.ExtrapWoa
   extrap_woa.Combine
   extrap_woa.Combine.setup
   extrap_woa.Combine.run
   extrap_woa.ExtrapStep
   extrap_woa.ExtrapStep.setup
   extrap_woa.ExtrapStep.run
   extrap_woa.RemapTopography
   extrap_woa.RemapTopography.setup
   extrap_woa.RemapTopography.constrain_resources
   extrap_woa.RemapTopography.run

ziso
~~~~

.. currentmodule:: compass.ocean.tests.ziso

.. autosummary::
   :toctree: generated/

   Ziso
   configure

   ZisoTestCase
   ZisoTestCase.configure
   ZisoTestCase.run

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

   mesh.cull.CullMeshStep
   mesh.cull.CullMeshStep.setup
   mesh.cull.CullMeshStep.run
   mesh.cull.cull_mesh

   mesh.floodplain.FloodplainMeshStep
   mesh.floodplain.FloodplainMeshStep.run

   mesh.remap_topography.RemapTopography
   mesh.remap_topography.RemapTopography.setup
   mesh.remap_topography.RemapTopography.constrain_resources
   mesh.remap_topography.RemapTopography.run


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
