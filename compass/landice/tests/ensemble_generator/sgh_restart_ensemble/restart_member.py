"""
Step for restarting a single incomplete ensemble member in-place.
"""

import os

from compass.step import Step


def _set_restart_in_namelist(namelist_path):
    """Set config_do_restart = .true. in-place in namelist.landice."""
    with open(namelist_path, 'r') as f:
        lines = f.readlines()

    updated = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('config_do_restart'):
            # Replace whatever value it has with .true., preserving any
            # leading whitespace on the original line.
            leading = line[:len(line) - len(line.lstrip())]
            new_lines.append(leading + 'config_do_restart = .true.\n')
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        # Append if not present
        new_lines.append('\nconfig_do_restart = .true.\n')

    with open(namelist_path, 'w') as f:
        f.writelines(new_lines)


class InPlaceRestartMember(Step):
    """
    A step for restarting an incomplete ensemble member in-place.

    Rather than copying files to a new directory, this step operates directly
    in the original run directory. It sets ``config_do_restart = .true.`` in
    ``namelist.landice`` so that the run continues from its last checkpoint
    when ``EnsembleManager`` calls ``sbatch job_script.sh`` using the
    original, unmodified job script.

    Attributes
    ----------
    run_num : int
        The run number for this ensemble member

    spinup_work_dir : str
        Path to the original spinup ensemble work directory
    """

    def __init__(self, test_case, run_num, spinup_work_dir):
        """
        Create an in-place restart step for an ensemble member

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        run_num : int
            The run number for this ensemble member

        spinup_work_dir : str
            Path to the directory containing the original spinup runs
        """
        self.run_num = run_num
        self.spinup_work_dir = spinup_work_dir

        name = f'run{run_num:03}_restart'
        super().__init__(test_case=test_case, name=name)

        # Override work_dir to point to the original run directory so that
        # EnsembleManager submits job_script.sh from the correct location.
        self.work_dir = os.path.join(spinup_work_dir, f'run{run_num:03}')

    def setup(self):
        """
        Prepare the original run directory for an in-place restart.

        ``self.work_dir`` is already set to the original spinup run directory
        by ``__init__``.  This method:

        1. Verifies the run directory and namelist.landice exist.
        2. Sets config_do_restart = .true. in namelist.landice.
        3. Creates a restart_attempt_N/ tracking directory.

        No files are copied or written, and no new job script is created.
        Job submission is handled by EnsembleManager using the original
        job_script.sh that was created when the spinup ensemble was set up.
        """
        run_dir = self.work_dir

        if not os.path.exists(run_dir):
            raise RuntimeError(
                f"Original run directory not found: {run_dir}")

        namelist_path = os.path.join(run_dir, 'namelist.landice')

        if not os.path.exists(namelist_path):
            raise RuntimeError(
                f"namelist.landice not found in {run_dir}")

        print(f'Setting config_do_restart = .true. in {namelist_path}')
        _set_restart_in_namelist(namelist_path)

        # Create a restart_attempt_N/ directory to track how many restarts
        # have been attempted for this run.  configure() counts these dirs
        # to enforce max_consecutive_restarts.  List the directory once to
        # find the highest existing attempt number, then create the next one.
        existing_nums = [
            int(d[len('restart_attempt_'):])
            for d in os.listdir(run_dir)
            if d.startswith('restart_attempt_') and
            d[len('restart_attempt_'):].isdigit()
        ]
        attempt_num = max(existing_nums, default=0) + 1
        attempt_dir = os.path.join(run_dir, f'restart_attempt_{attempt_num}')
        os.makedirs(attempt_dir, exist_ok=True)
        print(f'Tracking restart attempt {attempt_num} in {attempt_dir}')

    # No run() method — EnsembleManager handles job submission via sbatch
