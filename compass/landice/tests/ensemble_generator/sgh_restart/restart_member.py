"""
Step for restarting a single incomplete ensemble member.
"""

import os
import shutil

from compass.io import symlink
from compass.job import write_job_script
from compass.model import run_model
from compass.step import Step


class RestartMember(Step):
    """
    A step for restarting an incomplete ensemble member from checkpoint.

    This step:
    1. Links to the original run's restart files
    2. Updates configuration for restart (timestamps, namelist)
    3. Sets up job script
    4. Runs the restart

    Attributes
    ----------
    run_num : int
        The run number for this ensemble member

    spinup_work_dir : str
        Path to the original spinup ensemble work directory

    restart_attempt : int
        Which restart attempt this is (1 = first, 2 = second, etc.)
    """

    def __init__(self, test_case, run_num, spinup_work_dir):
        """
        Create a restart step for an ensemble member

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
        self.name = f'run{run_num:03}_restart'

        super().__init__(test_case=test_case, name=self.name)

    def setup(self):
        """
        Set up this restart by:
        1. Identifying the restart attempt number
        2. Copying necessary files from original run
        3. Updating restart configuration
        4. Setting up job script
        """

        print(f'Setting up restart for run number {self.run_num}')

        config = self.config
        run_name = f'run{self.run_num:03}'
        original_run_dir = os.path.join(self.spinup_work_dir, run_name)

        if not os.path.exists(original_run_dir):
            raise RuntimeError(
                f"Original run directory not found: {original_run_dir}")

        # Determine restart attempt number
        self.restart_attempt = self._get_restart_attempt_number(
            original_run_dir)
        restart_subdir = os.path.join(
            self.work_dir, f'restart_attempt_{
                self.restart_attempt}')
        os.makedirs(restart_subdir, exist_ok=True)

        # Read restart timestamp to determine simulation state
        restart_timestamp_file = os.path.join(
            original_run_dir, 'restart_timestamp')
        if not os.path.exists(restart_timestamp_file):
            raise RuntimeError(f"No restart_timestamp in {original_run_dir}")

        with open(restart_timestamp_file, 'r') as f:
            restart_time = f.read().strip()

        print(f"  {run_name}: Restarting from timestamp {restart_time}")
        print(f"  {run_name}: Restart attempt {self.restart_attempt}")

        # Copy essential configuration files
        files_to_copy = [
            'namelist.landice',
            'streams.landice',
            'albany_input.yaml',
            'run_info.cfg'
        ]

        for fname in files_to_copy:
            src = os.path.join(original_run_dir, fname)
            dst = os.path.join(restart_subdir, fname)
            if os.path.exists(src):
                shutil.copy(src, dst)

        # Add model as input
        self.add_model_as_input()

        # Copy or symlink restart file from original run
        self._copy_restart_file(original_run_dir, restart_subdir, restart_time)

        # Copy graph file if it exists
        graph_file = os.path.join(original_run_dir, 'graph.info')
        if os.path.exists(graph_file):
            shutil.copy(graph_file, restart_subdir)

        # Set up job script
        self.ntasks = config.getint('ensemble', 'ntasks', fallback=128)
        self.min_tasks = self.ntasks

        config.set('job', 'job_name', f'uq_{run_name}_r{self.restart_attempt}')
        machine = config.get('deploy', 'machine')

        # Create pre/post run commands
        pre_run_cmd = (
            'LOGDIR=restart_logs_`date +"%Y-%m-%d_%H-%M-%S"`;'
            'mkdir -p $LOGDIR; cp log* $LOGDIR 2>/dev/null || true; '
            'date'
        )
        post_run_cmd = "date"

        write_job_script(config, machine,
                         target_cores=self.ntasks, min_cores=self.min_tasks,
                         work_dir=restart_subdir,
                         pre_run_commands=pre_run_cmd,
                         post_run_commands=post_run_cmd)

        # Create symlink to load script if available
        if 'LOAD_COMPASS_ENV' in os.environ:
            script_filename = os.environ['LOAD_COMPASS_ENV']
            symlink(script_filename, os.path.join(restart_subdir,
                                                  'load_compass_env.sh'))

        # Store for run method
        self.restart_work_dir = restart_subdir
        self.original_run_dir = original_run_dir

    def run(self):
        """
        Run this restart of the ensemble member.
        """
        print(
            f"Running restart for run{
                self.run_num:03} (attempt {
                self.restart_attempt})")
        run_model(self)

    def _get_restart_attempt_number(self, original_run_dir):
        """
        Determine which restart attempt this is.

        Parameters
        ----------
        original_run_dir : str
            Directory of the original run

        Returns
        -------
        int
            Restart attempt number (1 for first restart, 2 for second, etc.)
        """
        # Count existing restart_attempt_* subdirectories
        restart_dirs = []
        if os.path.exists(original_run_dir):
            restart_dirs = [d for d in os.listdir(original_run_dir)
                            if d.startswith('restart_attempt_')]

        return len(restart_dirs) + 1

    def _copy_restart_file(
            self,
            original_run_dir,
            restart_subdir,
            restart_time):
        """
        Copy the appropriate restart file to the restart directory.

        Parameters
        ----------
        original_run_dir : str
            Directory of the original run

        restart_subdir : str
            Directory for this restart attempt

        restart_time : str
            Time string from restart_timestamp (format: YYYY-MM-DD_HH:MM:SS)
        """
        import glob

        # MALI restart files typically named as: rst.YYYY-MM-DD.nc
        # Extract just the date part from restart_time
        date_part = restart_time.split('_')[0]  # YYYY-MM-DD

        # Look for restart file with this date in output directory
        output_dir = os.path.join(original_run_dir, 'output')
        if os.path.exists(output_dir):
            pattern = os.path.join(output_dir, f'rst.{date_part}*.nc')
            restart_files = glob.glob(pattern)

            if restart_files:
                # Use the most recent (last) restart file
                src_file = sorted(restart_files)[-1]
                dst_file = os.path.join(restart_subdir, 'rst.restart.nc')
                shutil.copy(src_file, dst_file)
                print(f"  Copied restart file: {os.path.basename(src_file)}")
                return

        # Look in run directory directly (older style)
        pattern = os.path.join(original_run_dir, f'rst.{date_part}*.nc')
        restart_files = glob.glob(pattern)

        if restart_files:
            src_file = sorted(restart_files)[-1]
            dst_file = os.path.join(restart_subdir, 'rst.restart.nc')
            shutil.copy(src_file, dst_file)
            print(f"  Copied restart file: {os.path.basename(src_file)}")
        else:
            print(
                f"  WARNING: No restart file found matching date {date_part}")
            print("    Searched patterns:")
            print(f"      {os.path.join(output_dir, f'rst.{date_part}*.nc')}")
            print(
                f"      {
                    os.path.join(
                        original_run_dir,
                        f'rst.{date_part}*.nc')}")
