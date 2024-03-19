import glob
import os
import shutil
import sys
from importlib import resources

from jinja2 import Template
from mpas_tools.logging import check_call
from mpas_tools.scrip.from_mpas import scrip_from_mpas

from compass.io import symlink
from compass.job import write_job_script
from compass.model import make_graph_file, run_model
from compass.step import Step


class SetUpExperiment(Step):
    """
    A step for setting up an ISMIP6 experiment

    Attributes
    ----------
    """

    def __init__(self, test_case, name, subdir, exp):
        """
        Set up a new experiment

        Parameters
        ----------

        exp : experiment
        """

        self.exp = exp

        super().__init__(test_case=test_case, name=name, subdir=subdir)

    def setup(self):  # noqa:C901

        print(f"    Setting up experiment {self.exp}")

        config = self.config
        section = config['ismip6_run_ais_2300']
        self.ntasks = section.getint('ntasks')
        self.min_tasks = self.ntasks
        mesh_res = section.getint('mesh_res')
        forcing_basepath = section.get('forcing_basepath')
        init_cond_path = section.get('init_cond_path')
        init_cond_fname = os.path.split(init_cond_path)[-1]
        melt_params_path = section.get('melt_params_path')
        melt_params_fname = os.path.split(melt_params_path)[-1]
        region_mask_path = section.get('region_mask_path')
        region_mask_fname = os.path.split(region_mask_path)[-1]
        calving_method = section.get('calving_method')
        use_face_melting = section.getboolean('use_face_melting')
        sea_level_model = section.getboolean('sea_level_model')

        if self.exp == 'hist':
            exp_fcg = 'ctrlAE'
        else:
            exp_fcg = self.exp

        # Define where to get namelist, streams, yaml, etc. templates
        # (in current package)
        resource_location = \
            'compass.landice.tests.ismip6_run.ismip6_ais_proj2300'

        # Define calving method
        if calving_method == 'von_mises':
            use_vM_calving = True
        elif calving_method == 'restore':
            use_vM_calving = False

        # Figure out if the forcing is in tier1 or tier2 subdir
        if 'exp' in self.exp:
            if int(self.exp[-2:]) >= 7:
                forcing_basepath = os.path.join(forcing_basepath,
                                                'tier2_experiments')
            else:
                forcing_basepath = os.path.join(forcing_basepath,
                                                'tier1_experiments')
        else:
            forcing_basepath = os.path.join(forcing_basepath,
                                            'tier1_experiments')

        # Copy files we'll need from local paths specified in cfg file
        if self.exp == 'hist':
            shutil.copy(init_cond_path, self.work_dir)
        shutil.copy(melt_params_path, self.work_dir)
        shutil.copy(region_mask_path, self.work_dir)

        # Find and copy correct forcing files
        smb_search_path = os.path.join(
            forcing_basepath, exp_fcg,
            '*smb*bare*.nc')
        fcgFileList = glob.glob(smb_search_path)
        if len(fcgFileList) == 1:
            smb_path = fcgFileList[0]
            smb_fname = os.path.split(smb_path)[-1]
            shutil.copy(smb_path, self.work_dir)
        else:
            sys.exit("ERROR: Did not find exactly one matching SMB file at "
                     f"{smb_search_path}: {fcgFileList}")

        tf_search_path = os.path.join(forcing_basepath, exp_fcg,
                                      '*TF_*.nc')
        fcgFileList = glob.glob(tf_search_path)
        if len(fcgFileList) == 1:
            tf_path = fcgFileList[0]
            tf_fname = os.path.split(tf_path)[-1]
            shutil.copy(tf_path, self.work_dir)
        else:
            sys.exit("ERROR: Did not find exactly one matching TF file at "
                     f"{tf_search_path}: {fcgFileList}")

        # copy calving mask files for exp11-14
        useCalvingMask = False
        if exp_fcg[-2:].isdigit():
            exp_num = int(exp_fcg[-2:])
            if exp_num >= 11:
                mask_search_path = os.path.join(
                    forcing_basepath, exp_fcg,
                    'Antarctica_8to30km_ice_shelf_collapse_mask_*.nc')
                fcgFileList = glob.glob(mask_search_path)
                if len(fcgFileList) == 1:
                    mask_path = fcgFileList[0]
                    mask_fname = os.path.split(mask_path)[-1]
                    shutil.copy(mask_path, self.work_dir)
                    useCalvingMask = True
                else:
                    sys.exit("ERROR: Did not find exactly one matching "
                             "calving mask file at "
                             f"{mask_search_path}: {fcgFileList}")

        # Make stream modifications based on files that were determined above
        stream_replacements = {'input_file_SMB_forcing': smb_fname,
                               'input_file_TF_forcing': tf_fname}
        if self.exp == 'hist':
            stream_replacements['input_file_init_cond'] = init_cond_fname
            stream_replacements['input_file_region_mask'] = region_mask_fname
            stream_replacements['input_file_melt_params'] = melt_params_fname
        else:
            stream_replacements['input_file_init_cond'] = \
                'USE_RESTART_FILE_INSTEAD'
            stream_replacements['input_file_region_mask'] = \
                'USE_RESTART_FILE_INSTEAD'
            stream_replacements['input_file_melt_params'] = \
                'USE_RESTART_FILE_INSTEAD'
        if self.exp in ['hist', 'ctrlAE']:
            stream_replacements['forcing_interval'] = 'initial_only'
        else:
            stream_replacements['forcing_interval'] = '0001-00-00_00:00:00'

        self.add_streams_file(
            resource_location,
            'streams.landice.template',
            out_name='streams.landice',
            template_replacements=stream_replacements)

        if useCalvingMask:
            mask_stream_replacements = {'input_file_calving_mask_forcing_name':
                                        mask_fname}
            self.add_streams_file(
                resource_location, 'streams.mask_calving',
                out_name='streams.landice',
                template_replacements=mask_stream_replacements)

        if use_vM_calving:
            vM_param_path = section.get('von_mises_parameter_path')
            vM_stream_replacements = {'input_file_VM_params': vM_param_path}
            self.add_streams_file(
                resource_location, 'streams.vM_params',
                out_name='streams.landice',
                template_replacements=vM_stream_replacements)

        if use_face_melting:
            self.add_streams_file(
                resource_location, 'streams.faceMelting',
                out_name='streams.landice')

        # Set up namelist and customize as needed
        self.add_namelist_file(
            resource_location, 'namelist.landice',
            out_name='namelist.landice')

        # set up pio options because we are not using compass run
        pio_stride = section.getint('pio_stride')
        assert self.ntasks % pio_stride == 0, \
            'pio_stride is not evenly divisble into ntasks'
        io_tasks = self.ntasks // pio_stride
        options = {'config_pio_stride': f'{pio_stride}',
                   'config_pio_num_iotasks': f'{io_tasks}'}
        self.add_namelist_options(options=options,
                                  out_name='namelist.landice')

        if self.exp == 'hist':
            options = {'config_do_restart': ".false.",
                       'config_start_time': "'2000-01-01_00:00:00'",
                       'config_stop_time': "'2015-01-01_00:00:00'"}
            self.add_namelist_options(options=options,
                                      out_name='namelist.landice')

        if use_vM_calving:
            options = {'config_calving': "'von_Mises_stress'",
                       'config_restore_calving_front': ".false.",
                       'config_floating_von_Mises_threshold_stress_source':
                       "'data'",
                       'config_grounded_von_Mises_threshold_stress_source':
                       "'data'"}
            self.add_namelist_options(options=options,
                                      out_name='namelist.landice')

        # Include facemelting if needed
        if use_face_melting:
            options = {
                'config_front_mass_bal_grounded': "'ismip6'",
                'config_use_3d_thermal_forcing_for_face_melt': '.true.'}
            self.add_namelist_options(options=options,
                                      out_name='namelist.landice')

        if useCalvingMask:
            options = {'config_calving': "'none'",
                       'config_apply_calving_mask': ".true."}
            self.add_namelist_options(options=options,
                                      out_name='namelist.landice')

        if sea_level_model:
            # get config options
            slm_input_ice = section.get('slm_input_ice')
            slm_input_earth = section.get('slm_input_earth')
            slm_input_others = section.get('slm_input_others')
            nglv = section.getint('nglv')

            # complete the full paths to the SLM inputs
            slm_input_ice = os.path.join(slm_input_ice,
                                         f'GL{nglv}/ice_noGrIS_GL{nglv}/')
            slm_input_others = os.path.join(slm_input_others,
                                            f'GL{nglv}/')
            # incorporate the SLM config in the landice namelist
            options = {'config_uplift_method': "'sealevelmodel'"}
            self.add_namelist_options(options=options,
                                      out_name='namelist.landice')

            # change the sealevel namelist
            template = Template(resources.read_text
                                (resource_location,
                                 'namelist.sealevel.template'))
            text = template.render(nglv=int(nglv), slm_input_ice=slm_input_ice,
                                   slm_input_earth=slm_input_earth,
                                   slm_input_others=slm_input_others)

            # write out the namelise.sealevel file
            file_slm_nl = os.path.join(self.work_dir, 'namelist.sealevel')
            with open(file_slm_nl, 'w') as handle:
                handle.write(text)

            # create SLM output paths
            os.makedirs(os.path.join(self.work_dir, 'OUTPUT_SLM/'),
                        exist_ok='True')
            os.makedirs(os.path.join(self.work_dir, 'ICELOAD_SLM/'),
                        exist_ok='True')

        # For all projection runs, symlink the restart file for the
        # historical run
        # don't symlink restart_timestamp or you'll have a mighty mess
        if not self.exp == 'hist':
            os.symlink(f"../hist_{mesh_res:02}/rst.2015-01-01.nc",
                       os.path.join(self.work_dir, 'rst.2015-01-01.nc'))
            with open(os.path.join(self.work_dir, "restart_timestamp"),
                      "w") as text_file:
                text_file.write("2015-01-01_00:00:00")

        # add the albany_input.yaml file
        self.add_input_file(
            filename='albany_input.yaml',
            package=resource_location,
            copy=True)

        # create graph file
        make_graph_file(mesh_filename=init_cond_path,
                        graph_filename=os.path.join(self.work_dir,
                                                    'graph.info'))

        # COMPASS does not create symlinks for the load script in step dirs,
        # so use the standard approach for creating that symlink in each
        # step dir.
        if 'LOAD_COMPASS_ENV' in os.environ:
            script_filename = os.environ['LOAD_COMPASS_ENV']
            # make a symlink to the script for loading the compass conda env.
            symlink(script_filename, os.path.join(self.work_dir,
                                                  'load_compass_env.sh'))

        # customize job script
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

        # link in exe
        self.add_model_as_input()

    def run(self):
        """
        Run this step of the test case
        """

        config = self.config
        logger = self.logger
        section = config['ismip6_run_ais_2300']
        sea_level_model = section.getboolean('sea_level_model')
        if sea_level_model:
            self._build_mapping_files(config, logger)

        run_model(step=self, namelist='namelist.landice',
                  streams='streams.landice')

    def _build_mapping_files(self, config, logger):
        """
        Build mapping files between the MALI mesh and the SLM grid.

        Parameters
        ----------
        config : compass.config.CompassConfigParser
            Configuration options for a ismip6 forcing test case

        logger : logging.Logger
            A logger for output from the step
        """

        config = self.config
        section = config['ismip6_run_ais_2300']
        init_cond_path = section.get('init_cond_path')
        nglv = section.getint('nglv')

        # create the mapping files
        # first create scrip files
        mali_scripfile = 'mali_scripfile.nc'
        slm_scripfile = f'slm_nglv{nglv}scripfile.nc'
        mali_meshfile = 'mali_meshfile_sphereLatLon.nc'

        # slm scripfile
        logger.info(f'creating scripfile for the SLM grid with '
                    f'{nglv} Gauss-Legendre points in latitude')

        args = ['ncremap',
                '-g', slm_scripfile,
                '-G',
                f'latlon={nglv},{2*int(nglv)}#lat_typ=gss#lat_drc=n2s']

        check_call(args, logger=logger)

        # mali scripfile
        # first have to adjust lat-lon values as if they are on sphere
        shutil.copy(init_cond_path, mali_meshfile)
        args = ['set_lat_lon_fields_in_planar_grid.py',
                '--file', mali_meshfile,
                '--proj', 'ais-bedmap2-sphere']

        check_call(args, logger=logger)

        logger.info('creating scrip file for the mali mesh')
        scrip_from_mpas(mali_meshfile, mali_scripfile)

        # create mapping file from MALI mesh to SLM grid
        logger.info('creating MALI -> SLM grid mapfile with bilinear method')

        parallel_executable = config.get("parallel", "parallel_executable")
        # split the parallel executable into constituents
        args = parallel_executable.split(' ')
        args.extend(['ESMF_RegridWeightGen',
                     '-s', mali_scripfile,
                     '-d', slm_scripfile,
                     '-w', 'mapfile_mali_to_slm.nc',
                     '-m', 'conserve',
                     '-i', '-64bit_offset', '--netcdf4',
                     '--src_regional'])

        check_call(args, logger)

        # create mapping file from SLM grid to MALI mesh
        logger.info('SLM -> MALI mesh mapfile with bilinear method')
        args = parallel_executable.split(' ')
        args.extend(['ESMF_RegridWeightGen',
                     '-s', slm_scripfile,
                     '-d', mali_scripfile,
                     '-w', 'mapfile_slm_to_mali.nc',
                     '-m', 'bilinear',
                     '-i', '-64bit_offset', '--netcdf4',
                     '--dst_regional'])

        check_call(args, logger)

        # remove the scripfiles and copied mesh file
        logger.info('Removing the temporary mesh and scripfiles...')
        os.remove(slm_scripfile)
        os.remove(mali_scripfile)
        os.remove(mali_meshfile)
