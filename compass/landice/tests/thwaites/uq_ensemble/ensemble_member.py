from compass.step import Step
from compass.model import make_graph_file, run_model
from compass.job import write_job_script
from compass.io import symlink
from importlib import resources

import numpy as np
import os

class EnsembleMember(Step):
    """
    A step for setting up a single ensemble member

    Attributes
    ----------
    run_num : integer
        the run number for this ensemble member

    name : str
        the name of the run being set up in this step

    ntasks : integer
        the number of parallel (MPI) tasks the step would ideally use
    """

    def __init__(self, test_case, run_num):
        """
        Creates a new run within an ensemble

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        run_num : integer
            the run number for this ensemble member
        """
        self.run_num = run_num

        # define step (run) name
        self.name=f'run{run_num:03}'

        super().__init__(test_case=test_case, name=self.name)

    def setup(self):
        """
        Set up this run:
        * set up baseline run configuration
        * modify parameters for this ensemble member based on
          externally provided parameter vector
        """

        self.ntasks = 32 # chosen for Cori Haswell

        # Set up base run configuration
        self.add_namelist_file(
            'compass.landice.tests.thwaites.uq_ensemble', 'namelist.landice')

        self.add_streams_file(
            'compass.landice.tests.thwaites.uq_ensemble', 'streams.landice')

        self.add_input_file(filename='albany_input.yaml',
                        package='compass.landice.tests.thwaites.uq_ensemble',
                        copy=True)

        self.add_model_as_input()

        # modify param values as needed for this ensemble member

        # Use pre-defined parameter vectors in a text file
        param_file_name = self.config.get('thwaites_uq',
                                          'param_vector_filename')
        with resources.open_text('compass.landice.tests.thwaites.uq_ensemble', param_file_name) as params_file:
            vm_thresh_vec, gamma0_vec = np.loadtxt(params_file, delimiter=',',
                                                   skiprows=1,
                                                   usecols=(1,2), unpack=True)

        # von Mises stress threshold
        #self.vM_value = np.random.uniform(150.0e3, 400.0e3)
        self.vM_value = vm_thresh_vec[self.run_num]
        options = {'config_grounded_von_Mises_threshold_stress':
                   f'{self.vM_value}',
                   'config_floating_von_Mises_threshold_stress':
                   f'{self.vM_value}'}

        # von Mises speed limit
        secYr = 3600.0 * 24.0 * 365.0
        self.clv_spd_lim_value = np.random.uniform(8000.0, 30000.0)
        options['config_calving_speed_limit'] = \
                f'{self.clv_spd_lim_value / secYr}'

        self.add_namelist_options(options=options,
                                  out_name='namelist.landice')


        # set job name to run number so it will get set in batch script
        self.config.set('job', 'job_name', f'uq_{self.name}')
        machine = self.config.get('deploy', 'machine')
        write_job_script(self.config, machine,
                         target_cores=self.ntasks, min_cores=self.min_tasks,
                         work_dir=self.work_dir)

        # COMPASS does not create symlinks for the load script in step dirs,
        # so use the standard approach for creating that symlink in each step dir.
        if 'LOAD_COMPASS_ENV' in os.environ:
            script_filename = os.environ['LOAD_COMPASS_ENV']
            # make a symlink to the script for loading the compass conda env.
            symlink(script_filename, os.path.join(self.work_dir,
                                                 'load_compass_env.sh'))

    def run(self):
        """
        Run this member of the ensemble.
        Eventually we want this function to handle restarts.
        """

        make_graph_file(mesh_filename='/global/project/projectdirs/piscees/MALI_projects/Thwaites_UQ/thwaites_4km_mesh_20221122/input_files/Thwaites.nc',
                        graph_filename='graph.info')
        run_model(self)
