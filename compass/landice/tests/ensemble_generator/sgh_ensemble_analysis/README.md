# SGH Ensemble Analysis

Analyzes completed ensemble runs to evaluate steady-state behavior and data compatibility.

## Overview

This test case processes a completed ensemble (spinup or restart) and produces:

- **Steady-state analysis**: Determines if each run reached equilibrium using water mass balance
- **Data compatibility**: Validates against observational constraints (if specularity data available)
- **Results summary**: JSON file categorizing all runs by completion status

## Quick Start

After your spinup ensemble completes:

```bash
# 1. Create analysis config
cat > analysis.cfg << 'EOF'
[ensemble_generator]
ensemble_template = sgh_ensemble

[analysis_ensemble]
ensemble_work_dir = /path/to/spinup_ensemble
config_file = /path/to/ensemble_generator.cfg
EOF

# 2. Run analysis
compass setup -t landice/ensemble_generator/sgh_analysis \
    -w /work/analysis \
    -f analysis.cfg
compass run -w /work/analysis

# 3. View results
cat /work/analysis/analysis_summary.json | python -m json.tool
```

## Configuration

All parameters go in a single `analysis_ensemble.cfg` file:

### Required Settings

```ini
[analysis_ensemble]
ensemble_work_dir = /path/to/completed/ensemble
config_file = /path/to/ensemble_generator.cfg
```

The `ensemble_work_dir` should be the actual ensemble work directory (e.g., `/work/spinup_ensemble` or `/work/restart_ensemble`).

### Optional: Steady-State Parameters

```ini
[steady_state]
# Rolling window size (years) for mass balance check
window_years = 10.0

# Relative imbalance threshold
# Steady state when: |input - output| / (|input| + |output|) < threshold
# Default 0.05 = 5% relative error
imbalance_threshold = 0.05

# Generate plots (subglacial_water_mass_balance.png, etc.)
plot_results = False
```

**Tuning `window_years`**:
- Larger values (20-30 yrs): Smoother, less sensitive to noise
- Smaller values (5-10 yrs): More responsive to recent changes
- Default (10 yrs): Good for most simulations

**Tuning `imbalance_threshold`**:
- Stricter (0.01): 1% relative imbalance → harder to achieve steady state
- Default (0.05): 5% relative imbalance → reasonable for geophysical models
- Looser (0.10): 10% relative imbalance → easier to achieve

### Optional: Validation Parameters

```ini
[validation]
# Balanced accuracy threshold for data compatibility
# Both east and west AIS must exceed this
# Range [0.0, 1.0], typical 0.65
balanced_accuracy_threshold = 0.65

# Path to specularity content TIFF file
# If None or file doesn't exist, validation is skipped
spec_tiff_file = /path/to/specularity_content.tif

# Generate validation plots
plot_validation = False
```

### Optional: Output Settings

```ini
[output]
# Directory name for results (relative to work_dir)
results_directory = analysis_results
```

## Understanding Results

### Output Files

**`analysis_summary.json`** (main output):
```json
{
  "timestamp": "2026-03-19T14:30:00",
  "ensemble_dir": "/work/spinup_ensemble",
  "total_runs": 50,
  "completed_runs": 25,
  "incomplete_runs": 25,
  "steady_state_runs": [0, 2, 5, 7, ...],
  "data_compatible_runs": [0, 5, 8, ...],
  "both_criteria_runs": [0, 5, ...],
  "restart_needed_runs": [1, 3, 4, ...],
  "analysis_parameters": {
    "steady_state": {...},
    "validation": {...}
  },
  "individual_results": {
    "0": {
      "steady_state": {...},
      "validation": {...}
    },
    ...
  }
}
```

### Console Output Example

```
ENSEMBLE ANALYSIS SUMMARY
Total runs: 50
  Completed: 25
  Incomplete: 25

Steady-state runs: 15/25 (60.0%)
  [0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35]

Data-compatible runs: 12/25 (48.0%)
  [0, 5, 8, 12, 15, 20, 22, 25, 30, 35, 40, 42]

Both criteria met: 10/25 (40.0%)
  [0, 5, 12, 15, 20, 25, 30, 35]

Runs needing restart: 10
  [1, 3, 4, 6, 9, 11, 13, 14, 18, 19]
```

## Analysis Criteria Explained

### Steady-State Detection

A run is at **steady state** when the water mass balance equation is approximately satisfied over a rolling time window:

```
Input (melt + channel_melt) ≈ Output (all discharge fluxes)
```

More precisely:
```
relative_imbalance = |Input - Output| / (|Input| + |Output|) < threshold
```

The analysis checks this condition over the final portion of the simulation. If satisfied, the run is at steady state.

**Why this matters**:
- Steady-state runs have reached equilibrium and won't improve much with more time
- Non-steady runs may benefit from restarting
- See `restart/` to schedule restarts for non-steady runs

### Data Compatibility Validation

If a specularity content TIFF file is provided, each run is validated by comparing:

- **Model prediction**: Simulated subglacial water thickness
- **Observations**: Radar specularity content (proxy for wetness)

The metric is **balanced accuracy (BA)**:
```
BA = 0.5 * (true_positive_rate + true_negative_rate)
```

Both east and west Antarctic regions must have BA ≥ threshold.

**Why this matters**:
- Not all steady-state runs match observations
- Data-compatible runs have both equilibrium AND observational support
- Runs meeting both criteria are highest confidence

## Workflow: Analysis → Restart → Re-analyze

Typical workflow for iterative ensemble refinement:

```bash
# ============================================================
# Iteration 1: Initial Spinup
# ============================================================

compass setup -t spinup_ensemble -w /work/ens1 -f spinup.cfg
compass run -w /work/ens1/spinup_ensemble
# ... wait for jobs (~hours to days depending on job queue)

# ============================================================
# Iteration 1: Analyze Results
# ============================================================

cat > /work/analysis1.cfg << 'EOF'
[ensemble_generator]
ensemble_template = sgh_ensemble

[analysis_ensemble]
ensemble_work_dir = /work/ens1/spinup_ensemble
config_file = /work/spinup.cfg
EOF

compass setup -t landice/ensemble_generator/sgh_analysis \
    -w /work/analysis1 -f /work/analysis1.cfg
compass run -w /work/analysis1

# Results: 50 runs, 25 completed
#          15 steady-state, 10 need restart

# ============================================================
# Iteration 2: Schedule & Run Restarts
# ============================================================

python3 << 'PYTHON'
from compass.landice.tests.ensemble_generator.sgh_restart import schedule_restarts

config, runs = schedule_restarts(
    '/work/analysis1/analysis_summary.json',
    '/work/restart_ens'
)
PYTHON

compass setup -t landice/ensemble_generator/sgh_restart \
    -w /work/restart -f /work/restart_ens/restart_ensemble.cfg
compass run -w /work/restart
# ... wait for restart jobs

# ============================================================
# Iteration 2: Re-analyze Restarts
# ============================================================

cat > /work/analysis2.cfg << 'EOF'
[ensemble_generator]
ensemble_template = sgh_ensemble

[analysis_ensemble]
ensemble_work_dir = /work/restart/sgh_restart_ensemble
config_file = /work/restart_ens/restart_ensemble.cfg
EOF

compass setup -t landice/ensemble_generator/sgh_analysis \
    -w /work/analysis2 -f /work/analysis2.cfg
compass run -w /work/analysis2

# Results: 10 restart jobs, 8 now steady-state, 2 need another restart

# ============================================================
# Final: Aggregate All Results
# ============================================================

python3 << 'PYTHON'
from compass.landice.tests.ensemble_generator.sgh_analysis import ResultsAggregator

agg = ResultsAggregator('/work')
results = agg.aggregate()
agg.print_summary(results)
agg.save_aggregated(results)

print(f"Final: {len(results['final_steady_state_runs'])}/50 at steady state")
print(f"       {len(results['final_data_compatible_runs'])}/50 data-compatible")
PYTHON
```

## Advanced Usage

### Custom Analysis Parameters

Tighten steady-state criteria:

```ini
[steady_state]
window_years = 20.0              # Longer window, more stable
imbalance_threshold = 0.01       # Stricter (1% vs 5%)
```

### Generating Plots

Enable plot generation:

```ini
[steady_state]
plot_results = True

[validation]
plot_validation = True
```

This creates:
- `subglacial_water_mass_balance.png`
- `water_mass_balance_residual.png`
- `subglacial_hydrology_timeseries.png`
- `spec_subglacialHydro_validation.png`

These help visualize model behavior and validation.

### Analyzing Specific Ensembles

Point to any completed ensemble work directory:

```ini
[analysis_ensemble]
# Can be spinup, restart, or branch
ensemble_work_dir = /work/any_completed_ensemble
config_file = /work/any_ensemble_generator.cfg
```

## Troubleshooting

### "No analysis results"

- The ensemble hasn't completed yet (check job queue)
- The ensemble_work_dir is wrong (should point to actual work dir with run000/, run001/, etc.)

### All runs marked "incomplete"

- Runs may have failed (check log files in run directories)
- Check that `output/globalStats.nc` exists for each run

### No validation results

- Specularity TIFF file not found or not specified
- Set `spec_tiff_file = None` to skip validation

### Plots not generated

- Enable with `plot_results = True` or `plot_validation = True`
- Check that matplotlib and cmocean are installed

## See Also

- `restart/`: Schedule and run restarts for non-steady runs
- `spinup/`: Initial ensemble setup and execution
- `branch/`: Branch from spinup for projection scenarios
