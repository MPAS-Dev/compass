landice
-------

.. currentmodule:: compass.landice

.. autosummary::
   :toctree: generated/

   Landice

Utilities
^^^^^^^^^

.. currentmodule:: compass.landice.util

.. autosummary::
   :toctree: generated/

   calculate_decomp_core_pair

ISMIP7 framework
^^^^^^^^^^^^^^^^

.. currentmodule:: compass.landice.ismip7

.. autosummary::
   :toctree: generated/

   ice_sheet_params.get_params
   mapping.build_mapping_file
   remap.extrapolate_source
   remap.open_rename_and_trim
   remap.add_xtime_and_write

Test Groups
^^^^^^^^^^^

antarctica
~~~~~~~~~~

.. currentmodule:: compass.landice.tests.antarctica

.. autosummary::
   :toctree: generated/

   Antarctica

   mesh_gen.MeshGen

   mesh.Mesh
   mesh.Mesh.run

calving_dt_convergence
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: compass.landice.tests.calving_dt_convergence

.. autosummary::
   :toctree: generated/

   CalvingDtConvergence

   dt_convergence_test.DtConvergenceTest
   dt_convergence_test.DtConvergenceTest.validate

   run_model.RunModel
   run_model.RunModel.run


circular_shelf
~~~~~~~~~~~~~~

.. currentmodule:: compass.landice.tests.circular_shelf

.. autosummary::
   :toctree: generated/

   CircularShelf

   decomposition_test.DecompositionTest
   decomposition_test.DecompositionTest.run

   run_model.RunModel
   run_model.RunModel.run

   setup_mesh.SetupMesh
   setup_mesh.SetupMesh.run

   visualize.Visualize
   visualize.Visualize.run
   visualize.visualize_circular_shelf


crane
~~~~~

.. currentmodule:: compass.landice.tests.crane

.. autosummary::
   :toctree: generated/

   Crane

   mesh_gen.MeshGen
   mesh_gen.MeshGen.run

   mesh.Mesh
   mesh.Mesh.run


dome
~~~~

.. currentmodule:: compass.landice.tests.dome

.. autosummary::
   :toctree: generated/

   Dome

   decomposition_test.DecompositionTest
   decomposition_test.DecompositionTest.run

   restart_test.RestartTest
   restart_test.RestartTest.run

   smoke_test.SmokeTest
   smoke_test.SmokeTest.run

   run_model.RunModel
   run_model.RunModel.setup
   run_model.RunModel.run

   setup_mesh.SetupMesh
   setup_mesh.SetupMesh.run

   visualize.Visualize
   visualize.Visualize.run
   visualize.visualize_dome


enthalpy_benchmark
~~~~~~~~~~~~~~~~~~

.. currentmodule:: compass.landice.tests.enthalpy_benchmark

.. autosummary::
   :toctree: generated/

   EnthalpyBenchmark

   A.A
   A.A.configure
   A.A.run
   A.visualize.Visualize
   A.visualize.Visualize.run

   B.B
   B.B.configure
   B.B.run
   B.visualize.Visualize
   B.visualize.Visualize.run

   run_model.RunModel
   run_model.RunModel.setup
   run_model.RunModel.run

   setup_mesh.SetupMesh
   setup_mesh.SetupMesh.run


eismint2
~~~~~~~~

.. currentmodule:: compass.landice.tests.eismint2

.. autosummary::
   :toctree: generated/

   Eismint2

   decomposition_test.DecompositionTest
   decomposition_test.DecompositionTest.run

   restart_test.RestartTest
   restart_test.RestartTest.run

   standard_experiments.StandardExperiments
   standard_experiments.StandardExperiments.run
   standard_experiments.visualize.Visualize
   standard_experiments.visualize.Visualize.run
   standard_experiments.visualize.visualize_eismint2

   run_experiment.RunExperiment
   run_experiment.RunExperiment.setup
   run_experiment.RunExperiment.run

   setup_mesh.SetupMesh
   setup_mesh.SetupMesh.run

ensemble_generator
~~~~~~~~~~~~~~~~~~

.. currentmodule:: compass.landice.tests.ensemble_generator

.. autosummary::
   :toctree: generated/

   EnsembleGenerator

   ensemble_manager.EnsembleManager
   ensemble_manager.EnsembleManager.setup
   ensemble_manager.EnsembleManager.run

   ensemble_member.EnsembleMember
   ensemble_member.EnsembleMember.setup
   ensemble_member.EnsembleMember.run

   ensemble_template.get_ensemble_template_name
   ensemble_template.get_spinup_template_package
   ensemble_template.get_branch_template_package

   spinup_ensemble.SpinupEnsemble
   spinup_ensemble.SpinupEnsemble.configure

   branch_ensemble.BranchEnsemble
   branch_ensemble.BranchEnsemble.configure

greenland
~~~~~~~~~

.. currentmodule:: compass.landice.tests.greenland

.. autosummary::
   :toctree: generated/

   Greenland

   decomposition_test.DecompositionTest
   decomposition_test.DecompositionTest.configure
   decomposition_test.DecompositionTest.run

   restart_test.RestartTest
   restart_test.RestartTest.run

   smoke_test.SmokeTest
   smoke_test.SmokeTest.run

   run_model.RunModel
   run_model.RunModel.setup
   run_model.RunModel.run

   mesh.Mesh
   mesh.Mesh.setup
   mesh.Mesh.run

   mesh_gen.MeshGen
   mesh_gen.MeshGen.run

humboldt
~~~~~~~~

.. currentmodule:: compass.landice.tests.humboldt

.. autosummary::
   :toctree: generated/

   Humboldt

   mesh_gen.MeshGen
   mesh_gen.MeshGen.run

   mesh.Mesh
   mesh.Mesh.run

   run_model.RunModel
   run_model.RunModel.run

   decomposition_test.DecompositionTest
   decomposition_test.DecompositionTest.configure
   decomposition_test.DecompositionTest.validate

   restart_test.RestartTest
   restart_test.RestartTest.configure
   restart_test.RestartTest.validate

hydro_radial
~~~~~~~~~~~~

.. currentmodule:: compass.landice.tests.hydro_radial

.. autosummary::
   :toctree: generated/

   HydroRadial

   decomposition_test.DecompositionTest
   decomposition_test.DecompositionTest.run

   restart_test.RestartTest
   restart_test.RestartTest.run

   spinup_test.SpinupTest
   spinup_test.SpinupTest.run

   steady_state_drift_test.SteadyStateDriftTest
   steady_state_drift_test.SteadyStateDriftTest.run

   run_model.RunModel
   run_model.RunModel.setup
   run_model.RunModel.run

   setup_mesh.SetupMesh
   setup_mesh.SetupMesh.run

   visualize.Visualize
   visualize.Visualize.run
   visualize.visualize_hydro_radial

ismip6_forcing
~~~~~~~~~~~~~~

.. currentmodule:: compass.landice.tests.ismip6_forcing

.. autosummary::
   :toctree: generated/

   Ismip6Forcing
   configure.configure
   create_mapfile.build_mapping_file
   create_mapfile.create_scrip_from_latlon

   atmosphere.Atmosphere
   atmosphere.Atmosphere.configure
   atmosphere.process_smb.ProcessSMB
   atmosphere.process_smb.ProcessSMB.setup
   atmosphere.process_smb.ProcessSMB.run
   atmosphere.process_smb.ProcessSMB.remap_ismip6_smb_to_mali
   atmosphere.process_smb.ProcessSMB.rename_ismip6_smb_to_mali_vars
   atmosphere.process_smb.ProcessSMB.correct_smb_anomaly_for_climatology
   atmosphere.process_smb_racmo.ProcessSmbRacmo
   atmosphere.process_smb_racmo.ProcessSmbRacmo.setup
   atmosphere.process_smb_racmo.ProcessSmbRacmo.run
   atmosphere.process_smb_racmo.ProcessSmbRacmo.remap_source_smb_to_mali
   atmosphere.process_smb_racmo.ProcessSmbRacmo.rename_source_smb_to_mali_vars
   atmosphere.process_smb_racmo.ProcessSmbRacmo.correct_smb_anomaly_for_base_smb

   ocean_basal.OceanBasal
   ocean_basal.OceanBasal.configure
   ocean_basal.process_basal_melt.ProcessBasalMelt
   ocean_basal.process_basal_melt.ProcessBasalMelt.setup
   ocean_basal.process_basal_melt.ProcessBasalMelt.run
   ocean_basal.process_basal_melt.ProcessBasalMelt.combine_ismip6_inputfiles
   ocean_basal.process_basal_melt.ProcessBasalMelt.remap_ismip6_basal_melt_to_mali_vars
   ocean_basal.process_basal_melt.ProcessBasalMelt.rename_ismip6_basal_melt_to_mali_vars

   ocean_thermal.OceanThermal
   ocean_thermal.OceanThermal.configure
   ocean_thermal.process_thermal_forcing.ProcessThermalForcing
   ocean_thermal.process_thermal_forcing.ProcessThermalForcing.setup
   ocean_thermal.process_thermal_forcing.ProcessThermalForcing.run
   ocean_thermal.process_thermal_forcing.ProcessThermalForcing.remap_ismip6_thermal_forcing_to_mali_vars
   ocean_thermal.process_thermal_forcing.ProcessThermalForcing.rename_ismip6_thermal_forcing_to_mali_vars

   shelf_collapse.ShelfCollapse
   shelf_collapse.ShelfCollapse.configure
   shelf_collapse.process_shelf_collapse.ProcessShelfCollapse.setup
   shelf_collapse.process_shelf_collapse.ProcessShelfCollapse.run
   shelf_collapse.process_shelf_collapse.ProcessShelfCollapse.remap_ismip6_shelf_mask_to_mali_vars
   shelf_collapse.process_shelf_collapse.ProcessShelfCollapse.rename_ismip6_shelf_mask_to_mali_vars

ismip6_run
~~~~~~~~~~

.. currentmodule:: compass.landice.tests.ismip6_run

.. autosummary::
   :toctree: generated/

   Ismip6Run

   ismip6_ais_proj2300.Ismip6AisProj2300
   ismip6_ais_proj2300.Ismip6AisProj2300.configure
   ismip6_ais_proj2300.Ismip6AisProj2300.run

   ismip6_ais_proj2300.set_up_experiment.SetUpExperiment
   ismip6_ais_proj2300.set_up_experiment.SetUpExperiment.setup
   ismip6_ais_proj2300.set_up_experiment.SetUpExperiment.run

ismip7_calibration
~~~~~~~~~~~~~~~~~~

.. currentmodule:: compass.landice.tests.ismip7_calibration

.. autosummary::
   :toctree: generated/

   Ismip7Calibration
   configure.check_options
   configure.melt_forms
   configure.objective_options
   configure.parameter_name
   configure.parameter_values
   configure.weighting
   datasets.climatology_files
   datasets.load_targets
   datasets.mask_files
   datasets.missing_files
   datasets.ocean_states
   objective.build_toolbox_terms
   objective.run_optimisation
   objective.scale_to_ensemble
   quadratic.angle_from_sin_slope
   quadratic.draft_slope
   quadratic.local_quadratic_melt
   quadratic.mean_slope
   quadratic.nonlocal_quadratic_melt
   quadratic.u_factor
   terms.average_by_group
   terms.calculate_term1
   terms.calculate_term2
   terms.calculate_term3
   terms.calculate_term4
   terms.integrate_by_group
   terms.stack_cells
   terms.uniform_area
   toolbox.check_integrity
   toolbox.file_sha256
   toolbox.toolbox_path

   replication.Replication
   replication.Replication.configure
   replication.Replication.validate
   replication.replicate.Replicate
   replication.replicate.Replicate.setup
   replication.replicate.Replicate.run
   replication.replicate.check_published

   ais.Ais
   ais.Ais.configure
   ais.Ais.validate
   ais.aggregate.Aggregate
   ais.aggregate.Aggregate.setup
   ais.aggregate.Aggregate.run
   ais.aggregate.unit_aggregates
   ais.aggregate.basin_coordinate
   ais.calibrate.Calibrate
   ais.calibrate.Calibrate.run
   ais.fit_delta_t.FitDeltaT
   ais.fit_delta_t.FitDeltaT.run
   ais.melt_model.basin_mean_tf
   ais.melt_model.initial_draft
   ais.melt_model.integrate_by_basin
   ais.melt_model.interpolate_to_draft
   ais.melt_model.melt_from_tf
   ais.melt_model.read_run
   ais.remap_forcing.RemapForcing
   ais.remap_forcing.RemapForcing.setup
   ais.remap_forcing.RemapForcing.run
   ais.remap_masks.RemapMasks
   ais.remap_masks.RemapMasks.setup
   ais.remap_masks.RemapMasks.run
   ais.report.Report
   ais.report.Report.run
   ais.run_state.RunState
   ais.run_state.RunState.setup
   ais.run_state.RunState.runtime_setup
   ais.run_state.RunState.run
   ais.verify_melt.VerifyMelt
   ais.verify_melt.VerifyMelt.setup
   ais.verify_melt.VerifyMelt.run

ismip7_forcing
~~~~~~~~~~~~~~

.. currentmodule:: compass.landice.tests.ismip7_forcing

.. autosummary::
   :toctree: generated/

   Ismip7Forcing
   configure.configure

   atmosphere.Atmosphere
   atmosphere.Atmosphere.configure
   atmosphere.process_smb.ProcessSmb
   atmosphere.process_smb.ProcessSmb.setup
   atmosphere.process_smb.ProcessSmb.run
   atmosphere.process_temperature.ProcessTemperature
   atmosphere.process_temperature.ProcessTemperature.setup
   atmosphere.process_temperature.ProcessTemperature.run
   atmosphere.process_smb_gradient.ProcessSmbGradient
   atmosphere.process_smb_gradient.ProcessSmbGradient.setup
   atmosphere.process_smb_gradient.ProcessSmbGradient.run
   atmosphere.process_temperature_gradient.ProcessTemperatureGradient
   atmosphere.process_temperature_gradient.ProcessTemperatureGradient.setup
   atmosphere.process_temperature_gradient.ProcessTemperatureGradient.run
   atmosphere.process_runoff.ProcessRunoff
   atmosphere.process_runoff.ProcessRunoff.setup
   atmosphere.process_runoff.ProcessRunoff.run

   ocean_thermal.OceanThermal
   ocean_thermal.OceanThermal.configure
   ocean_thermal.process_thermal_forcing.ProcessThermalForcing
   ocean_thermal.process_thermal_forcing.ProcessThermalForcing.setup
   ocean_thermal.process_thermal_forcing.ProcessThermalForcing.run

   fracture.Fracture
   fracture.Fracture.configure
   fracture.process_excess_melt.ProcessExcessMelt
   fracture.process_excess_melt.ProcessExcessMelt.setup
   fracture.process_excess_melt.ProcessExcessMelt.run
   fracture.process_lake_properties.ProcessLakeProperties
   fracture.process_lake_properties.ProcessLakeProperties.setup
   fracture.process_lake_properties.ProcessLakeProperties.run
   fracture.process_shelf_collapse.ProcessShelfCollapse
   fracture.process_shelf_collapse.ProcessShelfCollapse.setup
   fracture.process_shelf_collapse.ProcessShelfCollapse.run

isunnguata_sermia
~~~~~~~~~~~~~~~~~

.. currentmodule:: compass.landice.tests.isunnguata_sermia

.. autosummary::
   :toctree: generated/

   IsunnguataSermia

   mesh_gen.MeshGen
   mesh_gen.MeshGen.run

   mesh.Mesh
   mesh.Mesh.run

kangerlussuaq
~~~~~~~~~~~~~

.. currentmodule:: compass.landice.tests.kangerlussuaq

.. autosummary::
   :toctree: generated/

   Kangerlussuaq

   mesh_gen.MeshGen
   mesh_gen.MeshGen.run

   mesh.Mesh
   mesh.Mesh.run

koge_bugt_s
~~~~~~~~~~~

.. currentmodule:: compass.landice.tests.koge_bugt_s

.. autosummary::
   :toctree: generated/

   KogeBugtS

   mesh_gen.MeshGen
   mesh_gen.MeshGen.run

   mesh.Mesh
   mesh.Mesh.run

mesh_convergence
~~~~~~~~~~~~~~~~

.. currentmodule:: compass.landice.tests.mesh_convergence

.. autosummary::
   :toctree: generated/

   MeshConvergence

   conv_test_case.ConvTestCase
   conv_test_case.ConvTestCase.configure
   conv_test_case.ConvTestCase.update_cores

   conv_init.ConvInit
   conv_init.ConvInit.run

   conv_analysis.ConvAnalysis

   forward.Forward
   forward.Forward.setup
   forward.Forward.constrain_resources
   forward.Forward.run
   forward.Forward.get_dt_duration

   halfar.Halfar
   halfar.Halfar.create_init
   halfar.Halfar.create_analysis

   halfar.init.Init
   halfar.init.Init.run

   halfar.analysis.Analysis
   halfar.analysis.Analysis.run
   halfar.analysis.Analysis.rmse

   horizontal_advection.HorizontalAdvection
   horizontal_advection.HorizontalAdvection.create_init
   horizontal_advection.HorizontalAdvection.create_analysis

   horizontal_advection.init.Init
   horizontal_advection.init.Init.run

   horizontal_advection.analysis.Analysis
   horizontal_advection.analysis.Analysis.run
   horizontal_advection.analysis.Analysis.rmse

   horizontal_advection_thickness.HorizontalAdvectionThickness
   horizontal_advection_thickness.HorizontalAdvectionThickness.create_init
   horizontal_advection_thickness.HorizontalAdvectionThickness.create_analysis

   horizontal_advection_thickness.init.Init
   horizontal_advection_thickness.init.Init.run

   horizontal_advection_thickness.analysis.Analysis
   horizontal_advection_thickness.analysis.Analysis.run
   horizontal_advection_thickness.analysis.Analysis.rmse

mesh_modifications
~~~~~~~~~~~~~~~~~~

.. currentmodule:: compass.landice.tests.mesh_modifications

.. autosummary::
   :toctree: generated/

   MeshModifications

   subdomain_extractor.SubdomainExtractor

   subdomain_extractor.extract_region.ExtractRegion
 
mismipplus
~~~~~~~~~~

.. currentmodule:: compass.landice.tests.mismipplus

.. autosummary::
   :toctree: generated/

   MISMIPplus

   smoke_test.SmokeTest
   smoke_test.SmokeTest.validate

   spin_up.SpinUp
   spin_up.SpinUp.configure
    
   setup_mesh.SetupMesh
   setup_mesh.SetupMesh.run
   
   setup_mesh.calculate_mesh_params
   setup_mesh.mark_cull_cells_for_MISMIP
   setup_mesh.center_trough
    
   run_model.RunModel
   run_model.RunModel.setup
   run_model.RunModel.constrain_resources
   run_model.RunModel.process_inputs_and_outputs
   run_model.RunModel.run

   tasks.get_ntasks_from_cell_count
   tasks.exact_cell_count
   tasks.approx_cell_count 

slm_circ_icesheet
~~~

.. currentmodule:: compass.landice.tests.slm_circ_icesheet

.. autosummary::
   :toctree: generated/

   SlmCircIcesheet

   mesh_convergence.MeshConvergenceTest
   mesh_convergence.MeshConvergenceTest.configure
   smoke_test.SmokeTest

   run_model.RunModel
   run_model.RunModel.setup
   run_model.RunModel.run

   setup_mesh.SetupMesh
   setup_mesh.SetupMesh.run

   visualize.Visualize
   visualize.Visualize.setup
   visualize.Visualize.run
   visualize.visualize_slm_circsheet

thwaites
~~~~~~~~

.. currentmodule:: compass.landice.tests.thwaites

.. autosummary::
   :toctree: generated/

   Thwaites

   decomposition_test.DecompositionTest
   decomposition_test.DecompositionTest.configure
   decomposition_test.DecompositionTest.run

   restart_test.RestartTest
   restart_test.RestartTest.run

   run_model.RunModel
   run_model.RunModel.setup
   run_model.RunModel.run

   mesh_gen.MeshGen
   mesh_gen.MeshGen.run

   mesh.Mesh
   mesh.Mesh.run

Landice Framework
^^^^^^^^^^^^^^^^^

.. currentmodule:: compass.landice

.. autosummary::
   :toctree: generated/

   ais_observations

   extrapolate.extrapolate_variable

   iceshelf_melt.calc_mean_TF

   mesh.add_bedmachine_thk_to_ais_gridded_data
   mesh.add_grid_imask_from_dst_scrip_hull
   mesh.build_dst_scrip_hull
   mesh.clean_up_after_interp
   mesh.clip_mesh_to_bounding_box
   mesh.plot_hull_diagnostic
   mesh.get_mesh_config_bounding_box
   mesh.get_optional_interp_datasets
   mesh.gridded_flood_fill
   mesh.interp_gridded2mali
   mesh.mpas_flood_fill
   mesh.preprocess_ais_data
   mesh.run_optional_interpolation
   mesh.set_rectangular_geom_points_and_edges
   mesh.set_cell_width
   mesh.subset_gridded_dataset_to_bounds
   mesh.get_dist_to_edge_and_gl
   mesh.build_cell_width
   mesh.build_mali_mesh
   mesh.make_region_masks
