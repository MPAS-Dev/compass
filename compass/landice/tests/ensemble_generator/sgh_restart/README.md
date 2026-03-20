i# SGH Ensemble Restart

Restarts incomplete ensemble members from checkpoints to reach steady state and/or data compatibility.

## Overview

This test case continues incomplete ensemble members from their last checkpoint. It:

1. **Identifies** runs that didn't reach steady state
2. **Verifies** they have sufficient progress (minimum simulation years)
3. **Schedules** continuations with automated job submission
4. **Tracks** restart attempts to prevent infinite loops

## Quick Start

After analyzing an ensemble with `sgh_ensemble_analysis`:

```bash
# 1. Schedule restarts (creates config file)
python3 << 'PYTHON'
from compass.landice.tests.ensemble_generator.ensemble_templates.sgh_ensemble.restart import schedule_restarts

config, runs = schedule_restarts(
    '/work/analysis/analysis_summary.json',
    '/work/restart_ens'
)
print(f"Identified {len(runs)} runs to restart")
PYTHON

# 2. Run restart ensemble
compass setup -t landice/ensemble_generator/sgh_restart_ensemble \
    -w /work/restart \
    -f /work/restart_ens/restart_ensemble.cfg
compass run -w /work/restart

# 3. Re-analyze to check progress
compass setup -t landice/ensemble_generator/sgh_ensemble_analysis \
    -w /work/analysis2 \
    -f /work/analysis2_config.cfg
compass run -w /work/analysis2
```

## Configuration

The restart config is **generated automatically** by `schedule_restarts()`, but you can also create one manually.

### Required Settings

```ini
[restart_ensemble]
spinup_work_dir = /path/to/original/spinup_ensemble
```

This should point to the directory containing `run000/`, `run001/`, etc. from the original ensemble.

### Tuning Parameters

```ini
[restart_ensemble]

# Maximum consecutive restart attempts per run
# Prevents infinite loops if a run keeps failing
max_consecutive_restarts = 3

# Minimum simulation years before restart
# Prevents restarting runs that haven't made progress
min_simulation_years_before_restart = 50.0

# Whether to auto-restart incomplete runs
# Set to False for manual control (requires config edits)
auto_restart_incomplete = True
```

**Tuning `min_simulation_years_before_restart`**:
- Lower (20-30 yrs): More frequent restarts, higher computational cost
- Default (50 yrs): Good balance for typical simulations
- Higher (100+ yrs): Fewer restarts, larger jumps in time

**Tuning `max_consecutive_restarts`**:
- Lower (2): Stops after 2 attempts, saves resources
- Default (3): 3 attempts = ~150+ years possible
- Higher (4-5): Allows more attempts, more expensive

## How Restarts Work

### Restart Files

Each restart is organized as:

```
spinup_ensemble/run003/
├── output/
│   ├── globalStats.nc        (original output)
│   ├── rst.2050-01-01.nc     (original checkpoint)
│   └── history.nc
├── restart_attempt_1/         (first restart)
│   ├── job_script.sh
│   ├── namelist.landice
│   ├── output/               (new output from restart)
│   └── rst.restart.nc
├── restart_attempt_2/         (second restart)
│   ├── job_script.sh
│   ├── output/
│   └── rst.restart.nc
└── restart_attempt_3/         (third restart)
    └── ...
```

Each restart:
- Reads the previous checkpoint
- Updates timestamps
- Continues to the original `stop_time`
- Saves output to separate directory

### Completion Detection

The restart process checks:
1. Does the run have output? → No restart if missing
2. Has it completed? → No restart if already finished
3. Is it at steady state? → No restart if already satisfied
4. Has it made progress? → No restart if too short
5. Too many attempts? → No restart if max exceeded

## Workflow: Identify → Schedule → Run → Re-analyze

### Step 1: Analyze Spinup Ensemble

```bash
compass setup -t landice/ensemble_generator/sgh_ensemble_analysis \
    -w /work/analysis1 -f spinup_analysis.cfg
compass run -w /work/analysis1
```

Output: `/work/analysis1/analysis_summary.json` with categorized runs

### Step 2: Identify Restarts

```bash
python3 << 'PYTHON'
from compass.landice.tests.ensemble_generator.ensemble_templates.sgh_ensemble.restart import schedule_restarts

config, runs = schedule_restarts(
    '/work/analysis1/analysis_summary.json',
    '/work/restart_ens',
    min_years=50.0,      # Don't restart runs shorter than 50 years
    max_attempts=3       # Max 3 restart attempts
)

if runs:
    print(f"Will restart {len(runs)} runs: {runs}")
else:
    print("No runs to restart!")
PYTHON
```

This generates `/work/restart_ens/restart_ensemble.cfg` with:
- `spinup_work_dir` pointing to original ensemble
- List of runs to restart
- All parameters configured

### Step 3: Set Up Restart Ensemble

```bash
compass setup -t landice/ensemble_generator/sgh_restart_ensemble \
    -w /work/restart \
    -f /work/restart_ens/restart_ensemble.cfg
```

This creates restart steps for each identified run.

### Step 4: Run Restarts

```bash
compass run -w /work/restart
```

Ensemble manager submits SLURM jobs for all restarts and monitors them.

### Step 5: Re-analyze Restarts

```bash
compass setup -t landice/ensemble_generator/sgh_ensemble_analysis \
    -w /work/analysis2 -f restart_analysis.cfg
compass run -w /work/analysis2
```

Where `restart_analysis.cfg` points to:
```ini
[analysis_ensemble]
ensemble_work_dir = /work/restart/sgh_restart_ensemble
config_file = /work/restart_ens/restart_ensemble.cfg
```

Check results:
```bash
cat /work/analysis2/analysis_summary.json | python -m json.tool
```

### Step 6: Iterate if Needed

If some runs still need restart:

```python
# Repeat steps 2-5 to schedule another round of restarts
config, runs = schedule_restarts(
    '/work/analysis2/analysis_summary.json',
    '/work/restart_ens2'
)
```

## Understanding Restart Decisions

### Why a Run is Restarted

✅ **Restarted if**:
- Has output but didn't complete
- Made sufficient progress (≥ min_simulation_years)
- Not at steady state yet
- Below max restart attempts

### Why a Run is NOT Restarted

❌ **Skipped if**:
- Already completed (`restart_timestamp == stop_time`)
- Already at steady state
- Too short (simulation < min_simulation_years)
- No output files found
- Max restart attempts reached
- No analysis results available

### Example Output

```
Identifying restart candidates...
  run000: Restart candidate (85.2 yrs, 0 attempts)
  run001: Already completed
  run002: Restart candidate (63.5 yrs, 1 attempt)
  run003: Too short (42.3 < 50.0 yrs)
  run004: Already at steady state
  ...

Runs to restart: 10
Already at steady-state: 15
Data compatible: 12
Both criteria met: 10
```

## Advanced Configuration

### Manual Restart Selection

To restart specific runs only, create config manually:

```ini
[restart_ensemble]
spinup_work_dir = /work/spinup_ensemble
# Restarts will be auto-detected from analysis results
```

And edit `/work/restart_ens/restart_ensemble.cfg` before setup if needed.

### Conservative Restarts

Require longer simulations before restart:

```ini
[restart_ensemble]
min_simulation_years_before_restart = 100.0  # Very conservative
max_consecutive_restarts = 2                   # Few attempts
```

### Aggressive Restarts

More frequent restarts:

```ini
[restart_ensemble]
min_simulation_years_before_restart = 30.0   # Frequent restarts
max_consecutive_restarts = 5                  # Many attempts
```

## Troubleshooting

### "No runs to restart"

- All runs are already complete or at steady state
- Check analysis results to confirm
- Run analysis on ensemble to find incomplete runs

### "Max restart attempts reached"

- Run has been restarted 3 times (or configured max)
- Check if the run has persistent issues:
  ```bash
  ls /work/spinup_ensemble/run003/restart_attempt_*/log.landice*
  ```
- May need to adjust parameters or investigate model failures

### Restart jobs not submitting

- Check that `spinup_work_dir` exists and has run directories
- Verify `ensemble_manager` step is configured
- Check compass logs for errors

### Output not being saved

- Check `/path/to/run/restart_attempt_N/output/`
- Ensure disk space available
- Check SLURM logs for job failures

## Monitoring Restarts

Track restart progress:

```bash
# Check restart attempt directories
for run_dir in /work/spinup_ensemble/run*; do
  run_name=$(basename $run_dir)
  attempts=$(ls -d $run_dir/restart_attempt_* 2>/dev/null | wc -l)
  echo "$run_name: $attempts restart attempts"
done

# Check job queue
squeue -u $USER | grep uq_run

# Monitor output
tail -f /work/restart/sgh_restart_ensemble/run*/restart_attempt_*/log.landice.*.log
```

## Restart Attempt Statistics

After completion, analyze restart success:

```python
import os
import json
from pathlib import Path

spinup_dir = Path('/work/spinup_ensemble')

for run_dir in sorted(spinup_dir.glob('run*')):
    attempts = len(list(run_dir.glob('restart_attempt_*')))

    # Check if now at steady state
    analysis_file = run_dir / 'analysis_results.json'
    if analysis_file.exists():
        with open(analysis_file) as f:
            results = json.load(f)
        ss = results.get('steady_state', {}).get('is_steady_state', False)
        print(f"{run_dir.name}: {attempts} attempts → {'STEADY' if ss else 'NOT STEADY'}")
```

## See Also

- `analysis/`: Analyze runs to identify restarts
- `spinup/`: Initial ensemble setup
- `branch/`: Branch from spinup for projection scenarios
