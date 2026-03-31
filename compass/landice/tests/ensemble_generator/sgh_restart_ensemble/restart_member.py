"""
Step for restarting a single incomplete ensemble member in-place.
"""

import configparser
import os

from compass.io import symlink
from compass.job import write_job_script
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
    ``namelist.landice`` and writes a new ``job_script.sh`` so that the run
    continues from its last checkpoint when ``EnsembleManager`` calls
    ``sbatch job_script.sh``.

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

        This method:
        1. Sets config_do_restart = .true. in namelist.landice
        2. Registers the MALI model as an input
        3. Writes a new job_script.sh for sbatch submission
        4. Symlinks load_compass_env.sh if available

        No files are copied and no new subdirectories are created.
        Job submission is handled by EnsembleManager.
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

        # Register MALI executable so compass knows this step needs the model
        self.add_model_as_input()

        # 128 matches a typical HPC node count; user can override via config
        try:
            self.ntasks = self.config.getint('ensemble', 'ntasks')
        except (configparser.NoOptionError, configparser.NoSectionError):
            self.ntasks = 128

        self.min_tasks = self.ntasks

        run_name = f'run{self.run_num:03}'
        self.config.set('job', 'job_name', f'uq_{run_name}_restart')
        machine = self.config.get('deploy', 'machine')
        write_job_script(self.config, machine,
                         target_cores=self.ntasks, min_cores=self.min_tasks,
                         work_dir=self.work_dir)

        if 'LOAD_COMPASS_ENV' in os.environ:
            script_filename = os.environ['LOAD_COMPASS_ENV']
            symlink(script_filename, os.path.join(self.work_dir,
                                                  'load_compass_env.sh'))

    # No run() method — EnsembleManager handles job submission via sbatch
