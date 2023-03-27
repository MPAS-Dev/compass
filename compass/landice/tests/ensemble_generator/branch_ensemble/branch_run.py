import os
import shutil

import netCDF4

import compass.namelist
from compass.io import symlink
from compass.job import write_job_script
from compass.model import run_model
from compass.step import Step


class BranchRun(Step):
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

    input_file_name : str
        name of the input file that was read from the config

    basal_fric_exp : float
        value of basal friction exponent to use

    mu_scale : float
        value to scale muFriction by

    stiff_scale : float
        value to scale stiffnessFactor by

    von_mises_threshold : float
        value of von Mises stress threshold to use

    calv_spd_lim : float
        value of calving speed limit to use

    gamma0 : float
        value of gamma0 to use in ISMIP6 ice-shelf basal melt param.

    deltaT : float
        value of deltaT to use in ISMIP6 ice-shelf basal melt param.
    """

    def __init__(self, test_case, run_num,
                 basal_fric_exp=None,
                 mu_scale=None,
                 stiff_scale=None,
                 von_mises_threshold=None,
                 calv_spd_lim=None,
                 gamma0=None,
                 deltaT=None):
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
        self.name = f'run{run_num:03}'

        super().__init__(test_case=test_case, name=self.name)

    def setup(self):
        """
        Set up this run by setting up a baseline run configuration
        and then modifying parameters for this ensemble member based on
        an externally provided unit parameter vector
        """

        print(f'Setting up run number {self.run_num}')

        config = self.config
        section = config['branch_ensemble']

        control_test_dir = section.get('control_test_dir')
        branch_year = section.getint('branch_year')

        ctrl_dir = os.path.join(os.path.join(control_test_dir, self.name))

        # copy over the following:
        # restart file - but change year
        rst_file = os.path.join(ctrl_dir, f'rst.{branch_year:04}-01-01.nc')
        shutil.copy(rst_file, os.path.join(self.work_dir,
                                           'rst.2015-01-01.nc'))
        f = netCDF4.Dataset(os.path.join(self.work_dir,
                                         'rst.2015-01-01.nc'), 'r+')
        xtime = f.variables['xtime']
        xtime[0, :] = list('2015-01-01_00:00:00'.ljust(64))
        f.close()

        # create restart_timestamp
        with open(os.path.join(self.work_dir, 'restart_timestamp'), 'w') as f:
            f.write('2015-01-01_00:00:00')

        # yaml file
        shutil.copy(os.path.join(ctrl_dir, 'albany_input.yaml'),
                    self.work_dir)

        # set up namelist
        options = {'config_do_restart': '.true.',
                   'config_start_time': "'file'",
                   'config_stop_time': "'2300-01-01_00:00:00'"}
        namelist = compass.namelist.ingest(os.path.join(ctrl_dir,
                                                        'namelist.landice'))
        namelist = compass.namelist.replace(namelist, options)
        compass.namelist.write(namelist, os.path.join(self.work_dir,
                                                      'namelist.landice'))

        # set up streams
        stream_replacements = {}
        TF_file_path = section.get('TF_file_path')
        stream_replacements['TF_file_path'] = TF_file_path
        SMB_file_path = section.get('SMB_file_path')
        stream_replacements['SMB_file_path'] = SMB_file_path
        strm_src = 'compass.landice.tests.ensemble_generator.branch_ensemble'
        self.add_streams_file(strm_src,
                              'streams.landice',
                              out_name='streams.landice',
                              template_replacements=stream_replacements)

        # copy run_info file
        shutil.copy(os.path.join(ctrl_dir, 'run_info.cfg'), self.work_dir)

        # copy graph files
        shutil.copy(os.path.join(ctrl_dir, 'graph.info'), self.work_dir)

        # save number of tasks to use
        # eventually compass could determine this, but for now we want
        # explicit control
        self.ntasks = self.config.getint('ensemble', 'ntasks')
        self.min_tasks = self.ntasks

        self.add_model_as_input()

        # set job name to run number so it will get set in batch script
        self.config.set('job', 'job_name', f'uq_{self.name}')
        machine = self.config.get('deploy', 'machine')
        write_job_script(self.config, machine,
                         target_cores=self.ntasks, min_cores=self.min_tasks,
                         work_dir=self.work_dir)

        # COMPASS does not create symlinks for the load script in step dirs,
        # so use the standard approach for creating that symlink in each
        # step dir.
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

        run_model(self)
