import configparser
import os
import shutil
from importlib import resources

import netCDF4
import yaml

from compass.io import symlink
from compass.job import write_job_script
from compass.landice.extrapolate import extrapolate_variable
from compass.model import make_graph_file, run_model
from compass.step import Step


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
                 meltflux=None,
                 deltaT=None):
        """
        Creates a new run within an ensemble

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        run_num : integer
            the run number for this ensemble member

        basal_fric_exp : float
            value of basal friction exponent to use

        mu_scale : float
            value to scale muFriction by

        stiff_scale : float
            value to scale stiffnessFactor by

        von_mises_threshold : float
            value of von Mises stress threshold to use
            assumes same value for grounded and floating ice

        calv_spd_lim : float
            value of calving speed limit to use

        gamma0 : float
            value of gamma0 to use in ISMIP6 ice-shelf basal melt param.

        deltaT : float
            value of deltaT to use in ISMIP6 ice-shelf basal melt param.
        """
        self.run_num = run_num

        # store assigned param values for this run
        self.basal_fric_exp = basal_fric_exp
        self.mu_scale = mu_scale
        self.stiff_scale = stiff_scale
        self.von_mises_threshold = von_mises_threshold
        self.calv_spd_lim = calv_spd_lim
        self.gamma0 = gamma0
        self.meltflux = meltflux
        self.deltaT = deltaT

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
        if os.path.exists(os.path.join(self.work_dir, 'namelist.landice')):
            print(f"WARNING: {self.work_dir} already created; skipping.  "
                  "Please remove the directory "
                  f"{self.work_dir} and execute "
                  "'compass setup' again to set this experiment up.")
            return

        resource_module = 'compass.landice.tests.ensemble_generator'

        # Get config for info needed for setting up simulation
        config = self.config
        section = config['ensemble']

        # Create a python config (not compass config) file
        # for run-specific info useful for analysis/viz
        # (Could put these in compass step cfg file, but may be safer to
        # not mess with it.)
        run_info_cfg = configparser.ConfigParser()
        run_info_cfg.add_section('run_info')
        run_info_cfg.set('run_info', 'run_num', f'{self.run_num}')
        # Below we set values for each parameter being used in this ensemble

        # save number of tasks to use
        # eventually compass could determine this, but for now we want
        # explicit control
        self.ntasks = section.getint('ntasks')
        self.min_tasks = self.ntasks

        # Set up base run configuration
        self.add_namelist_file(resource_module, 'namelist.landice')

        # copy over albany yaml file
        # cannot use add_input functionality because need to modify the file
        # in this function, and inputs don't get processed until after this
        # function
        with resources.path(resource_module,
                            'albany_input.yaml') as package_path:
            target = str(package_path)
            shutil.copy(target, self.work_dir)

        self.add_model_as_input()

        # modify param values as needed for this ensemble member
        options = {}

        # CFL fraction is set from the cfg to force the user to make a
        # conscious decision about its value
        self.cfl_fraction = section.getfloat('cfl_fraction')
        options['config_adaptive_timestep_CFL_fraction'] = \
            f'{self.cfl_fraction}'

        # von Mises stress threshold
        if self.von_mises_threshold is not None:
            options['config_grounded_von_Mises_threshold_stress'] = \
                f'{self.von_mises_threshold}'
            options['config_floating_von_Mises_threshold_stress'] = \
                f'{self.von_mises_threshold}'
            run_info_cfg.set('run_info', 'von_mises_threshold',
                             f'{self.von_mises_threshold}')

        # calving speed limit
        if self.calv_spd_lim is not None:
            options['config_calving_speed_limit'] = \
                f'{self.calv_spd_lim}'
            run_info_cfg.set('run_info', 'calv_spd_limit',
                             f'{self.calv_spd_lim}')

        # adjust basal friction exponent
        # rename and copy base file
        input_file_path = section.get('input_file_path')
        input_file_name = input_file_path.split('/')[-1]
        base_fname = input_file_name.split('.')[:-1][0]
        new_input_fname = f'{base_fname}_MODIFIED.nc'
        self.input_file_name = new_input_fname  # store for run method
        shutil.copy(input_file_path, os.path.join(self.work_dir,
                                                  new_input_fname))
        # set input filename in streams and create streams file
        stream_replacements = {'input_file_init_cond': new_input_fname}
        if self.basal_fric_exp is not None:
            # adjust mu and exponent
            orig_fric_exp = section.getfloat('orig_fric_exp')
            _adjust_friction_exponent(orig_fric_exp, self.basal_fric_exp,
                                      os.path.join(self.work_dir,
                                                   new_input_fname),
                                      os.path.join(self.work_dir,
                                                   'albany_input.yaml'))
            run_info_cfg.set('run_info', 'basal_fric_exp',
                             f'{self.basal_fric_exp}')

        # mu scale
        if self.mu_scale is not None:
            f = netCDF4.Dataset(os.path.join(self.work_dir, new_input_fname),
                                'r+')
            f.variables['muFriction'][:] *= self.mu_scale
            f.close()
            run_info_cfg.set('run_info', 'mu_scale', f'{self.mu_scale}')

        # stiff scale
        if self.stiff_scale is not None:
            f = netCDF4.Dataset(os.path.join(self.work_dir, new_input_fname),
                                'r+')
            f.variables['stiffnessFactor'][:] *= self.stiff_scale
            f.close()
            run_info_cfg.set('run_info', 'stiff_scale', f'{self.stiff_scale}')

        # adjust gamma0 and deltaT
        # (only need to check one of these params)
        basal_melt_param_file_path = section.get('basal_melt_param_file_path')
        basal_melt_param_file_name = basal_melt_param_file_path.split('/')[-1]
        base_fname = basal_melt_param_file_name.split('.')[:-1][0]
        new_fname = f'{base_fname}_MODIFIED.nc'
        shutil.copy(basal_melt_param_file_path,
                    os.path.join(self.work_dir, new_fname))
        _adjust_basal_melt_params(os.path.join(self.work_dir, new_fname),
                                  self.gamma0, self.deltaT)
        stream_replacements['basal_melt_param_file_name'] = new_fname
        if self.gamma0 is not None:
            run_info_cfg.set('run_info', 'gamma0', f'{self.gamma0}')
        if self.deltaT is not None:
            run_info_cfg.set('run_info', 'meltflux', f'{self.meltflux}')
            run_info_cfg.set('run_info', 'deltaT', f'{self.deltaT}')

        # set up forcing files (unmodified)
        TF_file_path = section.get('TF_file_path')
        stream_replacements['TF_file_path'] = TF_file_path
        SMB_file_path = section.get('SMB_file_path')
        stream_replacements['SMB_file_path'] = SMB_file_path

        # store accumulated namelist and streams options
        self.add_namelist_options(options=options,
                                  out_name='namelist.landice')
        self.add_streams_file(resource_module, 'streams.landice',
                              out_name='streams.landice',
                              template_replacements=stream_replacements)

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

        # save run info for analysis/viz
        with open(os.path.join(self.work_dir, 'run_info.cfg'), 'w') \
             as run_config_file:
            run_info_cfg.write(run_config_file)

    def run(self):
        """
        Run this member of the ensemble.
        Eventually we want this function to handle restarts.
        """

        make_graph_file(mesh_filename=self.input_file_name,
                        graph_filename='graph.info')
        run_model(self)


def _adjust_friction_exponent(orig_fric_exp, new_fric_exp, filename,
                              albany_input_yaml):
    """
    Function to adjust the basal friction parameter by adjusting muFriction
    field in an input file to maintain the same basal shear stress, and
    then set the desired exponent value in the albany_input.yaml file.
    """
    f = netCDF4.Dataset(filename, 'r+')
    f.set_auto_mask(False)
    mu = f.variables['muFriction'][0, :]
    uX = f.variables['uReconstructX'][0, :, -1]
    uY = f.variables['uReconstructY'][0, :, -1]
    spd = (uX**2 + uY**2)**0.5 * (60. * 60. * 24. * 365.)
    mu = mu * spd**(orig_fric_exp - new_fric_exp)
    # The previous step leads to infs or nans in ice-free areas.
    # Set them all to 0.0 for the extrapolation step
    mu[spd == 0.0] = 0.0
    f.variables['muFriction'][0, :] = mu[:]
    f.close()

    extrapolate_variable(filename, 'muFriction', 'min')

    # now set exp in albany yaml file
    with open(albany_input_yaml, 'r') as stream:
        try:
            loaded = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    # Change value
    (loaded['ANONYMOUS']['Problem']['LandIce BCs']['BC 0']
        ['Basal Friction Coefficient']
        ['Power Exponent']) = float(new_fric_exp)
    # write out again
    with open(albany_input_yaml, 'w') as stream:
        try:
            yaml.dump(loaded, stream, default_flow_style=False)
        except yaml.YAMLError as exc:
            print(exc)


def _adjust_basal_melt_params(filename, gamma0=None, deltaT=None):
    """
    Function to adjust gamma0 and deltaT in a MALI input file for basal melt
    parameters.
    Currently assumes this is a regional mesh, and there is a single value of
    deltaT that should be used everywhere in the domain.
    """
    f = netCDF4.Dataset(filename, 'r+')
    f.set_auto_mask(False)
    if gamma0 is not None:
        f.variables['ismip6shelfMelt_gamma0'][:] = gamma0
    if deltaT is not None:
        f.variables['ismip6shelfMelt_deltaT'][:] = deltaT
    f.close()
