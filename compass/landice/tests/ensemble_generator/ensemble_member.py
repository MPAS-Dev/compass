import configparser
import os
import shutil
from importlib import resources

import netCDF4
import numpy as np

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

    gamma0 : float
        value of gamma0 to use in ISMIP6 ice-shelf basal melt param.

    deltaT : float
        value of deltaT to use in ISMIP6 ice-shelf basal melt param.
    """

    def __init__(self, test_case, run_num,
                 resource_module,
                 namelist_option_values=None,
                 namelist_parameter_values=None,
                 basal_fric_exp=None,
                 mu_scale=None,
                 stiff_scale=None,
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

        resource_module : str
            Package containing configuration-specific namelist, streams,
            and albany input files

        namelist_option_values : dict, optional
            A dictionary of namelist option names and values to be
            overridden for this ensemble member

        namelist_parameter_values : dict, optional
            A dictionary of run-info parameter names and values that
            correspond to entries in ``namelist_option_values``

        basal_fric_exp : float
            value of basal friction exponent to use

        mu_scale : float
            value to scale muFriction by

        stiff_scale : float
            value to scale stiffnessFactor by

        gamma0 : float
            value of gamma0 to use in ISMIP6 ice-shelf basal melt param.

        deltaT : float
            value of deltaT to use in ISMIP6 ice-shelf basal melt param.
        """
        self.run_num = run_num
        self.resource_module = resource_module
        if namelist_option_values is None:
            namelist_option_values = {}
        if namelist_parameter_values is None:
            namelist_parameter_values = {}
        self.namelist_option_values = dict(namelist_option_values)
        self.namelist_parameter_values = dict(namelist_parameter_values)

        # store assigned param values for this run
        self.basal_fric_exp = basal_fric_exp
        self.mu_scale = mu_scale
        self.stiff_scale = stiff_scale
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

        resource_module = self.resource_module

        # Get config for info needed for setting up simulation
        config = self.config
        section = config['ensemble_generator']
        spinup_section = config['spinup_ensemble']

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

        # albany_input.yaml is optional unless fric_exp perturbations are used.
        albany_input_name = 'albany_input.yaml'
        albany_input_path = os.path.join(self.work_dir, albany_input_name)
        albany_source = resources.files(resource_module).joinpath(
            albany_input_name)
        # Materialize a real filesystem path in case the package is not
        # directly on the filesystem (e.g., zip/loader-backed).
        with resources.as_file(albany_source) as albany_source_path:
            has_albany_input = albany_source_path.is_file()
            if has_albany_input:
                shutil.copy(str(albany_source_path), self.work_dir)

        self.add_model_as_input()

        # modify param values as needed for this ensemble member
        options = {}

        # CFL fraction is set from the cfg to force the user to make a
        # conscious decision about its value
        self.cfl_fraction = section.getfloat('cfl_fraction')
        options['config_adaptive_timestep_CFL_fraction'] = \
            f'{self.cfl_fraction}'

        # apply generic namelist float parameter perturbations
        for option_name, value in self.namelist_option_values.items():
            options[option_name] = f'{value}'
        for parameter_name, value in self.namelist_parameter_values.items():
            run_info_cfg.set('run_info', parameter_name, f'{value}')

        # adjust basal friction exponent
        # rename and copy base file
        input_file_path = spinup_section.get('input_file_path')
        input_file_name = input_file_path.split('/')[-1]
        base_fname = ".".join(input_file_name.split('.')[:-1])
        new_input_fname = f'{base_fname}_MODIFIED_run{self.run_num:03}.nc'
        self.input_file_name = new_input_fname  # store for run method
        shutil.copy(input_file_path, os.path.join(self.work_dir,
                                                  new_input_fname))
        # set input filename in streams and create streams file
        stream_replacements = {'input_file_init_cond': new_input_fname}
        if config.has_option('spinup_ensemble', 'fric_samples_file'):
            fric_samples_file = spinup_section.get('fric_samples_file')
            fric_map_file = spinup_section.get('fric_map_file')
            mpas_cellid_file = spinup_section.get('mpas_cellid_file')
            fric_sample_scale = spinup_section.getfloat('fric_sample_scale')
            _apply_fric_sample(
                sample_num=self.run_num,
                sample_file=fric_samples_file,
                mu_opt_file=fric_map_file,
                mpas_cellid_file=mpas_cellid_file,
                ic_file=os.path.join(self.work_dir, new_input_fname),
                scaling=fric_sample_scale)
            run_info_cfg.set('run_info', 'fric_sample_num', f'{self.run_num}')
        if self.basal_fric_exp is not None:
            if not has_albany_input:
                raise ValueError(
                    "Parameter 'fric_exp' requires 'albany_input.yaml' "
                    f"in template package '{resource_module}'.")
            # adjust mu and exponent
            orig_fric_exp = spinup_section.getfloat('orig_fric_exp')
            _adjust_friction_exponent(orig_fric_exp, self.basal_fric_exp,
                                      os.path.join(self.work_dir,
                                                   new_input_fname),
                                      albany_input_path)
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

        # adjust gamma0 and deltaT (if a basal melt parameter file is used)
        if config.has_option('spinup_ensemble', 'basal_melt_param_file_path'):
            basal_melt_param_file_path = spinup_section.get(
                'basal_melt_param_file_path')
            basal_melt_param_file_name = \
                basal_melt_param_file_path.split('/')[-1]
            base_fname = ".".join(basal_melt_param_file_name.split('.')[:-1])
            new_fname = f'{base_fname}_MODIFIED_run{self.run_num:03}.nc'
            shutil.copy(basal_melt_param_file_path,
                        os.path.join(self.work_dir, new_fname))
            _adjust_basal_melt_params(os.path.join(self.work_dir, new_fname),
                                      self.gamma0, self.deltaT)
            stream_replacements['basal_melt_param_file_name'] = new_fname
        elif any(value is not None for value in
                 (self.gamma0, self.meltflux, self.deltaT)):
            raise ValueError(
                "Parameters 'gamma0' and/or 'meltflux' require "
                "'basal_melt_param_file_path' in [spinup_ensemble].")

        if self.gamma0 is not None:
            run_info_cfg.set('run_info', 'gamma0', f'{self.gamma0}')
        if self.deltaT is not None:
            run_info_cfg.set('run_info', 'meltflux', f'{self.meltflux}')
            run_info_cfg.set('run_info', 'deltaT', f'{self.deltaT}')

        # set up forcing files (unmodified)
        TF_file_path = spinup_section.get('TF_file_path')
        stream_replacements['TF_file_path'] = TF_file_path
        SMB_file_path = spinup_section.get('SMB_file_path')
        stream_replacements['SMB_file_path'] = SMB_file_path
        if config.has_option('spinup_ensemble', 'runoff_file_path'):
            stream_replacements['runoff_file_path'] = spinup_section.get(
                'runoff_file_path')

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
    # using text replacement rather than yaml parser because yaml
    # parser can sometimes mangle Albany yaml files due to formatting not
    # being fully yaml-compliant.  It also writes out the yaml in an
    # arbitrary ordering, which is inconvenient.
    with open(albany_input_yaml, 'r') as file:
        yamldata = file.readlines()
    for linenum in range(len(yamldata)):
        line = yamldata[linenum]
        if 'Power Exponent' in line:
            yamldata[linenum] = (line.split(':')[0] +
                                 f': {float(new_fric_exp)}\n')
    with open(albany_input_yaml, 'w') as file:
        file.writelines(yamldata)


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


def _apply_fric_sample(sample_num, sample_file, mu_opt_file,
                       mpas_cellid_file, ic_file, scaling):
    """Apply a precomputed friction sample to ``muFriction`` in IC file."""
    samples = np.load(sample_file)
    sample = samples[sample_num, :]

    log_mu_opt = np.loadtxt(mu_opt_file)
    log_mu_opt = log_mu_opt[1:]  # skip first row containing array size

    mpas_map = np.loadtxt(mpas_cellid_file, dtype='int')
    mpas_map = mpas_map[1:]  # skip first row containing array size

    f = netCDF4.Dataset(ic_file, 'r+')
    f.set_auto_mask(False)
    n_cells = len(f.dimensions['nCells'])

    # Set baseline mu outside of the Albany inverse region.
    mu = 0.2 * np.ones(n_cells)
    mu[mpas_map[:] - 1] = np.exp(scaling * sample + log_mu_opt)

    f.variables['muFriction'][0, :] = mu[:]
    f.close()
