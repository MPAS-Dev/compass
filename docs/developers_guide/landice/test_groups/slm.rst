.. _dev_landice_slm:

slm
===

The ``slm`` test group (:py:class:`compass.landice.tests.slm.Slm`) adds an
framework for evaluating coupled MALI and Sea-Level Model (SLM)
behavior.  The group currently contains one test case,
``circular_icesheet_test``
(:py:class:`compass.landice.tests.slm.circ_icesheet.CircIcesheetTest`),
which is designed to compare MALI-derived and SLM-derived estimates of sea
level change for a controlled circular ice-sheet geometry.

framework
---------

The shared config options for this group are documented in
:ref:`landice_slm` in the User's Guide.

The ``configure()`` method of ``CircIcesheetTest`` parses two comma-delimited
parameter lists from the config:

* ``mali_res`` from the ``[circ_icesheet]`` section, and
* ``slm_nglv`` from the ``[slm]`` section.

The test case then creates one ``setup_mesh`` and one ``run_model`` step for
each ``(mali_res, slm_nglv)`` pair.  Step names follow the pattern
``mali<res>km_slm<nglv>/<step>``.  A single ``visualize`` step is then added
to summarize and compare all runs.

setup_mesh
~~~~~~~~~~

The class
:py:class:`compass.landice.tests.slm.circ_icesheet.setup_mesh.SetupMesh`
builds all inputs required by MALI and SLM coupling for each
``(mali_res, slm_nglv)`` combination:

* creates a planar hex mesh and culls it,
* converts the mesh to a MALI grid and writes ``graph.info``,
* applies idealized circular initial conditions,
* writes an SMB forcing file (either radial or vertical forcing), and
* builds MALI<->SLM mapping files with ESMF.

The mapping workflow creates an SLM SCRIP grid from ``slm_nglv`` and
``slm_res``, creates a MALI SCRIP file from the generated MALI grid, and then
builds both ``mapping_file_mali_to_slm.nc`` and
``mapping_file_slm_to_mali.nc``.

run_model
~~~~~~~~~

The class
:py:class:`compass.landice.tests.slm.circ_icesheet.run_model.RunModel`
consumes the mesh, graph, SMB forcing, and mapping files from ``setup_mesh``
and runs MALI with SLM coupling options enabled in ``namelist.landice``.

During ``setup()``, the step also:

* creates ``OUTPUT_SLM`` and ``ICELOAD_SLM`` directories,
* renders ``namelist.sealevel`` from the Jinja template
  ``compass.landice.tests.slm.namelist.sealevel.template``, and
* fills SLM input paths from the ``slm_input_root`` config option.

The expected structure beneath ``slm_input_root`` is:

* ``icemodel/GL<nglv>/ice_noGrIS_GL<nglv>/``
* ``others/GL<nglv>/``
* ``earthmodel/``

visualize
~~~~~~~~~

The class
:py:class:`compass.landice.tests.slm.circ_icesheet.visualize.Visualize`
collects all ``output.nc`` files produced by the parameter sweep and computes
diagnostic quantities for coupled MALI-SLM comparison.

For each run combination, the analysis computes grounded-ice mass change,
sea-level contributions with and without the ``z0`` term, and percent-error
metrics relative to SLM outputs.  The step can generate:

* per-case time-series plots,
* per-resolution error curves over ``slm_nglv``, and
* a heat map of mean percent error.

The ``[circ_icesheet_viz]`` section controls plot output (save/hide figures)
and provides constant ocean-area values used as analysis fallbacks.