import glob
import os
import sys
from importlib import resources

from jinja2 import Template

from compass.job import write_job_script
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

    def setup(self):  # noqa: C901
        """
        Set up the experiment directory with all needed files.
        """
        print(f"    Setting up experiment {self.exp}")

        config = self.config
        section = config['ismip7_run_ais']
        self.ntasks = section.getint('ntasks')
        self.min_tasks = self.ntasks
        forcing_basepath = section.get('forcing_basepath')
        init_cond_path = section.get('init_cond_path')
        init_cond_fname = os.path.split(init_cond_path)[-1]
        melt_params_path = section.get('melt_params_path')
        melt_params_fname = os.path.split(melt_params_path)[-1]
        region_mask_path = section.get('region_mask_path')
        region_mask_fname = os.path.split(region_mask_path)[-1]
        calving_method = section.get('calving_method')
        sea_level_model = section.getboolean('sea_level_model')

        exp_info = self.exp_info
        scenario = exp_info['scenario']
        model = exp_info['model']
        is_historical = exp_info['is_historical']
        start_time = exp_info['start_time']
        stop_time = exp_info['stop_time']

        # Define where to get templates (in current package)
        resource_location = 'compass.landice.tests.ismip7_run.ismip7_ais'

        # Define calving method
        use_vM_calving = (calving_method == 'von_mises')

        # --- Determine forcing file paths ---
        if scenario == 'ocx':
            ocx_forcing_path = section.get('ocx_forcing_path')
            forcing_dir = ocx_forcing_path
        elif scenario == 'ctrl':
            # Control run uses climatology forcing
            forcing_dir = None  # handled separately below
        else:
            # Standard ESM-driven experiment
            forcing_dir = os.path.join(forcing_basepath,
                                       f"{model}_{scenario}")

        # --- Symlink input files ---
        if is_historical:
            os.symlink(init_cond_path,
                       os.path.join(self.work_dir,
                                    os.path.basename(init_cond_path)))
        os.symlink(melt_params_path,
                   os.path.join(self.work_dir,
                                os.path.basename(melt_params_path)))
        os.symlink(region_mask_path,
                   os.path.join(self.work_dir,
                                os.path.basename(region_mask_path)))

        # --- Find and symlink forcing files ---
        if scenario == 'ctrl':
            # Control run: use climatology files
            ctrl_tf_path = section.get('ctrl_tf_climatology_path')
            ctrl_atm_path = section.get('ctrl_atm_climatology_path')
            tf_fname = os.path.split(ctrl_tf_path)[-1]
            os.symlink(ctrl_tf_path,
                       os.path.join(self.work_dir, tf_fname))

            # Find atmosphere climatology files
            smb_files = glob.glob(os.path.join(ctrl_atm_path, '*SMB*.nc'))
            smb_files = [f for f in smb_files if 'gradient' not in f]
            if len(smb_files) == 1:
                smb_fname = os.path.split(smb_files[0])[-1]
                os.symlink(smb_files[0],
                           os.path.join(self.work_dir, smb_fname))
            else:
                sys.exit(f"ERROR: Expected 1 SMB climatology file in "
                         f"{ctrl_atm_path}, found {len(smb_files)}")

            temp_files = glob.glob(
                os.path.join(ctrl_atm_path, '*temperature*.nc'))
            temp_files = [f for f in temp_files if 'gradient' not in f]
            if len(temp_files) == 1:
                temp_fname = os.path.split(temp_files[0])[-1]
                os.symlink(temp_files[0],
                           os.path.join(self.work_dir, temp_fname))
            else:
                sys.exit(f"ERROR: Expected 1 temperature climatology file in "
                         f"{ctrl_atm_path}, found {len(temp_files)}")

            runoff_files = glob.glob(
                os.path.join(ctrl_atm_path, '*runoff*.nc'))
            if len(runoff_files) == 1:
                runoff_fname = os.path.split(runoff_files[0])[-1]
                os.symlink(runoff_files[0],
                           os.path.join(self.work_dir, runoff_fname))
            else:
                runoff_fname = ''  # runoff may not exist for ctrl

            smb_grad_files = glob.glob(
                os.path.join(ctrl_atm_path, '*SMB_gradient*.nc'))
            smb_grad_fname = ''
            if len(smb_grad_files) == 1:
                smb_grad_fname = os.path.split(smb_grad_files[0])[-1]
                os.symlink(smb_grad_files[0],
                           os.path.join(self.work_dir, smb_grad_fname))

            temp_grad_files = glob.glob(
                os.path.join(ctrl_atm_path, '*temperature_gradient*.nc'))
            temp_grad_fname = ''
            if len(temp_grad_files) == 1:
                temp_grad_fname = os.path.split(temp_grad_files[0])[-1]
                os.symlink(temp_grad_files[0],
                           os.path.join(self.work_dir, temp_grad_fname))

        else:
            # Standard or OCX experiment: find forcing in forcing_dir
            atm_dir = os.path.join(forcing_dir, 'atmosphere')
            ocean_dir = os.path.join(forcing_dir, 'ocean_thermal_forcing')

            # SMB forcing
            smb_search = os.path.join(atm_dir, '*SMB_*.nc')
            smb_list = glob.glob(smb_search)
            smb_list = [f for f in smb_list if 'gradient' not in f]
            if len(smb_list) == 1:
                smb_fname = os.path.split(smb_list[0])[-1]
                os.symlink(smb_list[0],
                           os.path.join(self.work_dir, smb_fname))
            else:
                sys.exit(f"ERROR: Expected 1 SMB file at {smb_search}, "
                         f"found {len(smb_list)}: {smb_list}")

            # Temperature forcing
            temp_search = os.path.join(atm_dir, '*temperature_*.nc')
            temp_list = glob.glob(temp_search)
            temp_list = [f for f in temp_list if 'gradient' not in f]
            if len(temp_list) == 1:
                temp_fname = os.path.split(temp_list[0])[-1]
                os.symlink(temp_list[0],
                           os.path.join(self.work_dir, temp_fname))
            else:
                sys.exit(f"ERROR: Expected 1 temperature file at "
                         f"{temp_search}, found {len(temp_list)}")

            # Runoff forcing (optional — may not exist for all experiments)
            runoff_search = os.path.join(atm_dir, '*runoff_*.nc')
            runoff_list = glob.glob(runoff_search)
            runoff_fname = ''
            if len(runoff_list) == 1:
                runoff_fname = os.path.split(runoff_list[0])[-1]
                os.symlink(runoff_list[0],
                           os.path.join(self.work_dir, runoff_fname))

            # SMB gradient (lapse rate)
            smb_grad_search = os.path.join(atm_dir, '*SMB_gradient_*.nc')
            smb_grad_list = glob.glob(smb_grad_search)
            smb_grad_fname = ''
            if len(smb_grad_list) == 1:
                smb_grad_fname = os.path.split(smb_grad_list[0])[-1]
                os.symlink(smb_grad_list[0],
                           os.path.join(self.work_dir, smb_grad_fname))

            # Temperature gradient (lapse rate)
            temp_grad_search = os.path.join(atm_dir,
                                            '*temperature_gradient_*.nc')
            temp_grad_list = glob.glob(temp_grad_search)
            temp_grad_fname = ''
            if len(temp_grad_list) == 1:
                temp_grad_fname = os.path.split(temp_grad_list[0])[-1]
                os.symlink(temp_grad_list[0],
                           os.path.join(self.work_dir, temp_grad_fname))

            # Thermal forcing
            tf_search = os.path.join(ocean_dir, '*thermal_forcing_*.nc')
            tf_list = glob.glob(tf_search)
            if len(tf_list) == 1:
                tf_fname = os.path.split(tf_list[0])[-1]
                os.symlink(tf_list[0],
                           os.path.join(self.work_dir, tf_fname))
            else:
                sys.exit(f"ERROR: Expected 1 TF file at {tf_search}, "
                         f"found {len(tf_list)}: {tf_list}")

        # --- Set up streams ---
        # Determine forcing interval
        if scenario == 'ctrl':
            forcing_interval_monthly = 'initial_only'
            forcing_interval_annual = 'initial_only'
        elif is_historical:
            # Historical: read forcing at each interval
            forcing_interval_monthly = '0000-01-00_00:00:00'
            forcing_interval_annual = '0001-00-00_00:00:00'
        else:
            # Projections
            forcing_interval_monthly = '0000-01-00_00:00:00'
            forcing_interval_annual = '0001-00-00_00:00:00'

        stream_replacements = {
            'input_file_init_cond': init_cond_fname if is_historical
            else 'USE_RESTART_FILE_INSTEAD',
            'input_file_region_mask': region_mask_fname if is_historical
            else 'USE_RESTART_FILE_INSTEAD',
            'input_file_melt_params': melt_params_fname,
            'input_file_SMB_forcing': smb_fname,
            'input_file_temperature_forcing': temp_fname,
            'input_file_TF_forcing': tf_fname,
            'input_file_runoff_forcing': runoff_fname,
            'input_file_smb_gradient_forcing': smb_grad_fname,
            'input_file_temperature_gradient_forcing': temp_grad_fname,
            'forcing_interval_monthly': forcing_interval_monthly,
            'forcing_interval_annual': forcing_interval_annual,
        }

        self.add_streams_file(
            resource_location,
            'streams.landice.template',
            out_name='streams.landice',
            template_replacements=stream_replacements)

        # --- Set up namelist ---
        self.add_namelist_file(
            resource_location, 'namelist.landice',
            out_name='namelist.landice')

        # PIO options
        pio_stride = section.getint('pio_stride')
        io_tasks = self.ntasks // pio_stride
        options = {'config_pio_stride': f'{pio_stride}',
                   'config_pio_num_iotasks': f'{io_tasks}'}
        self.add_namelist_options(options=options,
                                  out_name='namelist.landice')

        # Historical-specific options
        if is_historical:
            options = {'config_do_restart': ".false.",
                       'config_start_time': f"'{start_time}'",
                       'config_stop_time': f"'{stop_time}'"}
            self.add_namelist_options(options=options,
                                      out_name='namelist.landice')
        else:
            options = {'config_stop_time': f"'{stop_time}'"}
            self.add_namelist_options(options=options,
                                      out_name='namelist.landice')

        # Calving options
        if use_vM_calving:
            vM_path = section.get('von_mises_parameter_path')
            options = {
                'config_calving': "'von_Mises_stress'",
                'config_restore_calving_front': ".false.",
                'config_floating_von_Mises_threshold_stress_source': "'data'",
                'config_grounded_von_Mises_threshold_stress_source': "'data'"}
            self.add_namelist_options(options=options,
                                      out_name='namelist.landice')
            vM_stream_replacements = {'input_file_VM_params': vM_path}
            self.add_streams_file(
                resource_location, 'streams.vM_params',
                out_name='streams.landice',
                template_replacements=vM_stream_replacements)

        # Sea-level model options
        if sea_level_model:
            slm_input_ice = section.get('slm_input_ice')
            slm_input_earth = section.get('slm_input_earth')
            slm_earth_structure = section.get('slm_earth_structure')
            slm_input_others = section.get('slm_input_others')
            nglv = section.getint('nglv')

            slm_input_ice = os.path.join(slm_input_ice,
                                         f'GL{nglv}/ice_noGrIS_GL{nglv}/')
            slm_input_others = os.path.join(slm_input_others,
                                            f'GL{nglv}/')

            options = {'config_uplift_method': "'sealevelmodel'"}
            self.add_namelist_options(options=options,
                                      out_name='namelist.landice')

            template = Template(resources.read_text(
                resource_location, 'namelist.sealevel.template'))
            text = template.render(
                nglv=int(nglv), slm_input_ice=slm_input_ice,
                slm_input_earth=slm_input_earth,
                slm_earth_structure=slm_earth_structure,
                slm_input_others=slm_input_others)

            file_slm_nl = os.path.join(self.work_dir, 'namelist.sealevel')
            with open(file_slm_nl, 'w') as handle:
                handle.write(text)

            os.makedirs(os.path.join(self.work_dir, 'OUTPUT_SLM/'),
                        exist_ok=True)
            os.makedirs(os.path.join(self.work_dir, 'ICELOAD_SLM/'),
                        exist_ok=True)

            map_dir = os.path.join('..', 'mapping_files')
            for map_file in ('mapfile_mali_to_slm.nc',
                             'mapfile_slm_to_mali.nc'):
                os.symlink(os.path.join(map_dir, map_file),
                           os.path.join(self.work_dir, map_file))

        # --- Symlink restart for projections/ctrl ---
        if not is_historical:
            hist_exp = f"historical_{model}"
            os.symlink(f"../{hist_exp}/rst.2015-01-01.nc",
                       os.path.join(self.work_dir, 'rst.2015-01-01.nc'))
            with open(os.path.join(self.work_dir, "restart_timestamp"),
                      "w") as text_file:
                text_file.write("2015-01-01_00:00:00")

        # --- Add albany yaml, graph file, load script, job script ---
        self.add_input_file(
            filename='albany_input.yaml',
            package=resource_location,
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
