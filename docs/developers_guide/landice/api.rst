landice
-------

.. currentmodule:: compass.landice

.. autosummary::
   :toctree: generated/

   Landice

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
   decomposition_test.DecompositionTest.run

   restart_test.RestartTest
   restart_test.RestartTest.run

   smoke_test.SmokeTest
   smoke_test.SmokeTest.run

   run_model.RunModel
   run_model.RunModel.setup
   run_model.RunModel.run

   mesh.Mesh
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
   decomposition_test.DecompositionTest.validate

   restart_test.RestartTest
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
   run_model.RunModel.run

   tasks.get_ntasks_from_cell_count
   tasks.exact_cell_count
   tasks.approx_cell_count 

thwaites
~~~~~~~~

.. currentmodule:: compass.landice.tests.thwaites

.. autosummary::
   :toctree: generated/

   Thwaites

   decomposition_test.DecompositionTest
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
   mesh.clean_up_after_interp
   mesh.gridded_flood_fill
   mesh.interp_gridded2mali
   mesh.mpas_flood_fill
   mesh.preprocess_ais_data
   mesh.set_rectangular_geom_points_and_edges
   mesh.set_cell_width
   mesh.get_dist_to_edge_and_gl
   mesh.build_cell_width
   mesh.build_mali_mesh
   mesh.make_region_masks
