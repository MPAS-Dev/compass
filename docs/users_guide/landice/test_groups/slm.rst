.. _landice_slm:

slm
===

The ``landice/slm`` test group adds a workflow for coupled
MALI and Sea-Level Model (SLM)
configurations.  It currently contains one test case,
``circular_icesheet_test``, that creates an idealized circular ice sheet and
compares MALI- and SLM-derived sea-level diagnostics across a configurable
sweep of mesh and SLM resolutions.

The test case is useful for:

* validating coupled MALI-SLM setup and mapping files,
* comparing sensitivity to MALI horizontal resolution and SLM ``nglv``, and
* generating summary error plots for sea-level diagnostics.

circular_icesheet_test
----------------------

The test case path is:

``landice/slm/circular_icesheet_test``

At configure time, the test case reads:

* ``mali_res`` from ``[circ_icesheet]`` and
* ``slm_nglv`` from ``[slm]``.

For each ``(mali_res, slm_nglv)`` combination, it creates:

* ``mali<res>km_slm<nglv>/setup_mesh``
* ``mali<res>km_slm<nglv>/run_model``

It also creates one shared ``visualize`` step to analyze all combinations.

setup_mesh
~~~~~~~~~~

``setup_mesh`` creates the model mesh and forcing inputs:

* builds and culls a planar hex mesh,
* converts the mesh to a MALI grid and writes ``graph.info``,
* sets circular initial thickness and bed topography,
* creates SMB forcing (``horizontal`` or ``vertical`` forcing mode), and
* generates MALI<->SLM mapping files.

Mapping files are generated with ESMF using methods from ``[slm]``:

* ``mapping_method_mali_to_slm`` and
* ``mapping_method_slm_to_mali``.

run_model
~~~~~~~~~

``run_model`` runs MALI with SLM coupling enabled and uses the files generated
in ``setup_mesh``:

* ``landice_grid.nc``
* ``graph.info``
* ``smb_forcing.nc``
* ``mapping_file_mali_to_slm.nc``
* ``mapping_file_slm_to_mali.nc``

The step writes ``namelist.sealevel`` from a template and creates
``OUTPUT_SLM`` and ``ICELOAD_SLM`` directories for SLM outputs.

SLM input root and file layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``[slm]`` option ``slm_input_root`` controls where static SLM inputs are
read from.  The expected subdirectory structure beneath this root is:

* ``icemodel/GL<nglv>/ice_noGrIS_GL<nglv>/``
* ``others/GL<nglv>/``
* ``earthmodel/``

For example:

.. code-block:: cfg

    [slm]
    slm_input_root = /path/to/SeaLevelModel_Inputs

visualize
~~~~~~~~~

The ``visualize`` step reads each run's ``output/output.nc`` and computes:

* grounded-ice mass and mass change,
* sea-level contributions using multiple correction terms, and
* percent-error metrics comparing MALI and SLM results.

When multiple ``mali_res`` and/or ``slm_nglv`` values are requested,
``visualize`` also creates summary plots including error curves and a heat map
of mean percent error.

config options
--------------

The test case uses four config sections:

* ``[circ_icesheet]``: domain size, ice geometry, and MALI resolution list
* ``[smb_forcing]``: forcing direction and temporal evolution
* ``[slm]``: SLM input root, ``slm_nglv`` list, SLM resolution, mapping options,
  and stride used for post-processing alignment
* ``[circ_icesheet_viz]``: plotting controls and fallback ocean-area constants

The defaults are in
``compass/landice/tests/slm/circ_icesheet/circ_icesheet.cfg``.