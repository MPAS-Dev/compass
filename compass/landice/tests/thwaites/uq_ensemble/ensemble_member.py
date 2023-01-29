from compass.step import Step
from compass.model import make_graph_file, run_model
from compass.job import write_job_script
from compass.io import symlink
from importlib import resources
from compass.landice.extrapolate import extrapolate_variable

import numpy as np
import os
import shutil
import subprocess
import yaml
import netCDF4

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

    def __init__(self, test_case, run_num,
                 basal_fric_exp=None,
                 von_mises_threshold=None,
                 calv_spd_lim=None):
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

        # store assigned param values for this run
        self.basal_fric_exp = basal_fric_exp
        self.von_mises_threshold = von_mises_threshold
        self.calv_spd_lim = calv_spd_lim

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

        print(f'Setting up run number {self.run_num}')
        print(f'    basal_fric_exp={self.basal_fric_exp}, von_mises_threshold={self.von_mises_threshold}, calv_spd_lim={self.calv_spd_lim}')

        self.ntasks = 32 # chosen for Cori Haswell

        # Set up base run configuration
        self.add_namelist_file(
            'compass.landice.tests.thwaites.uq_ensemble', 'namelist.landice')

        self.add_streams_file(
            'compass.landice.tests.thwaites.uq_ensemble', 'streams.landice')

        # copy over albany yaml file
        # cannot use add_input functionality because we need to modify the file
        # in this function, and inputs don't get processed until after this
        # function
        with resources.path('compass.landice.tests.thwaites.uq_ensemble',
                            'albany_input.yaml') as package_path:
            target = str(package_path)
            shutil.copy(target, self.work_dir)

        self.add_model_as_input()

        # copy in input file so it can be modified
        config = self.config
        section = config['thwaites_uq']
        input_file_path = section.get('input_file_path')
        shutil.copy(input_file_path, self.work_dir)
        input_file_name = input_file_path.split('/')[-1]

        # modify param values as needed for this ensemble member

        # von Mises stress threshold
        options = {'config_grounded_von_Mises_threshold_stress':
                   f'{self.von_mises_threshold}',
                   'config_floating_von_Mises_threshold_stress':
                   f'{self.von_mises_threshold}'}

        # calving speed limit
        options['config_calving_speed_limit'] = \
                f'{self.calv_spd_lim}'

        # adjust basal friction exponent
        orig_fric_exp = 0.2  # set by initial condition file being used
        _adjust_friction_exponent(orig_fric_exp, self.basal_fric_exp,
                                  os.path.join(self.work_dir, input_file_name),
                                  os.path.join(self.work_dir,
                                               'albany_input.yaml'))

        # store modified namelist options
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


def _adjust_friction_exponent(orig_fric_exp, new_fric_exp, filename, albany_input_yaml):
    f = netCDF4.Dataset(filename, 'r+')
    f.set_auto_mask(False)
    mu = f.variables['muFriction'][0,:]
    uX = f.variables['uReconstructX'][0,:,-1]
    uY = f.variables['uReconstructY'][0,:,-1]
    spd = (uX**2 + uY**2)**0.5 * (60.*60.*24.*365.)
    mu = mu * spd**(orig_fric_exp - new_fric_exp)
    mu[spd == 0.0] = 0.0  # The previous step leads to infs or nans in ice-free areas. Set them all to 0.0 for the extrapolation step
    f.variables['muFriction'][0,:] = mu[:]
    f.close()

    extrapolate_variable(filename, 'muFriction', 'min')

    # now set exp in albany yaml file
    with open(albany_input_yaml, 'r') as stream:
        try:
            loaded = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    # Change value
    loaded['ANONYMOUS']['Problem']['LandIce BCs']['BC 0']['Basal Friction Coefficient']['Power Exponent'] = float(new_fric_exp)
    # write out again
    with open(albany_input_yaml, 'w') as stream:
        try:
            yaml.dump(loaded, stream, default_flow_style=False)
        except yaml.YAMLError as exc:
            print(exc)
