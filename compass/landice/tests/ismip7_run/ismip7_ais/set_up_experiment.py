import os
import sys

from compass.job import write_job_script
from compass.landice.tests.ismip7_run.ismip7_ais.stage_experiment import (
    configure_experiment_namelist_and_streams,
    symlink_experiment_inputs,
)
from compass.load_script import symlink_load_script
from compass.model import make_graph_file, run_model
from compass.step import Step


class SetUpExperiment(Step):
    """
    A step for setting up an ISMIP7 AIS experiment
    """

    def __init__(self, test_case, name, subdir, exp, exp_info):
        """
        Set up a new experiment

        Parameters
        ----------
        test_case : compass.testcase.TestCase
            The test case this step belongs to

        name : str
            The name of this step (same as the experiment name)

        subdir : str
            Subdirectory for this step

        exp : str
            Experiment identifier (e.g., 'ssp585_CESM2-WACCM')

        exp_info : dict
            Dictionary with experiment metadata:
            scenario, model, start_time, stop_time, is_historical
        """
        self.exp = exp
        self.exp_info = exp_info

        super().__init__(test_case=test_case, name=name, subdir=subdir)

    def setup(self):
        """
        Set up the experiment directory with all needed files.
        """
        print(f"    Setting up experiment {self.exp}")

        config = self.config
        section = config['ismip7_run_ais']
        self.ntasks = section.getint('ntasks')
        self.min_tasks = self.ntasks
        init_cond_path = section.get('init_cond_path')

        exp_info = self.exp_info

        # --- Symlink input files (init cond, forcing, melt params, region
        # mask, reference surface, and, for projections/ctrl, the restart
        # file from the corresponding historical run) ---
        filenames = symlink_experiment_inputs(self, self.exp, exp_info)

        # --- Set up namelist and streams files ---
        configure_experiment_namelist_and_streams(
            self, exp_info, filenames, out_name='landice')

        # --- Add albany yaml, graph file, load script, job script ---
        self.add_input_file(
            filename='albany_input.yaml',
            package='compass.landice.tests.ismip7_run.ismip7_ais',
            copy=True)

        make_graph_file(mesh_filename=init_cond_path,
                        graph_filename=os.path.join(self.work_dir,
                                                    'graph.info'))

        symlink_load_script(self.work_dir)

        self.config.set('job', 'job_name', self.exp)
        machine = self.config.get('deploy', 'machine')
        pre_run_cmd = ('LOGDIR=previous_logs_`date +"%Y-%m-%d_%H-%M-%S"`;'
                       'mkdir $LOGDIR; cp log* $LOGDIR; date')
        post_run_cmd = "date"
        write_job_script(self.config, machine,
                         target_cores=self.ntasks, min_cores=self.min_tasks,
                         work_dir=self.work_dir,
                         pre_run_commands=pre_run_cmd,
                         post_run_commands=post_run_cmd)

        self.add_model_as_input()

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        section = config['ismip7_run_ais']
        sea_level_model = section.getboolean('sea_level_model')
        if sea_level_model:
            map_dir = os.path.join('..', 'mapping_files')
            for map_file in ('mapfile_mali_to_slm.nc',
                             'mapfile_slm_to_mali.nc'):
                if not os.path.isfile(os.path.join(map_dir, map_file)):
                    sys.exit(f"ERROR: 'mapping_files/{map_file}' "
                             "does not exist in workdir. "
                             "Please run the 'mapping_files' step "
                             "before proceeding.")

        run_model(step=self, namelist='namelist.landice',
                  streams='streams.landice')
