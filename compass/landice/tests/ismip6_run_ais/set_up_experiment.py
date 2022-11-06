import os
from compass.model import run_model, make_graph_file
from compass.step import Step
import shutil, glob, sys
from jinja2 import Template
from importlib import resources


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

    def setup(self):

        print(f"    Setting up experiment {self.exp}")

        if self.exp == 'hist':
            exp_fcg = 'ctrlAE'
        else:
            exp_fcg = self.exp

        config = self.config
        section = config['ismip6_run_ais']
        mesh_res = section.get('mesh_res')
        compass_load_path = section.get('compass_load_path')
        forcing_basepath = section.get('forcing_basepath')
        init_cond_path = section.get('init_cond_path')
        init_cond_fname = os.path.split(init_cond_path)[-1]
        melt_params_path = section.get('melt_params_path')
        melt_params_fname = os.path.split(melt_params_path)[-1]
        region_mask_path = section.get('region_mask_path')
        region_mask_fname = os.path.split(region_mask_path)[-1]
        graph_files_path = section.get('graph_files_path')

        # Figure out if the forcing is in tier1 or tier2 subdir
        if 'exp' in self.exp:
            if int(self.exp[-2:]) >= 7:
                forcing_basepath = os.path.join(forcing_basepath, 'tier2_experiments')
            else:
                forcing_basepath = os.path.join(forcing_basepath, 'tier1_experiments')
        else:
            forcing_basepath = os.path.join(forcing_basepath, 'tier1_experiments')

        # Copy files we'll need from local paths specified in cfg file
        if self.exp == 'hist':
            shutil.copy(init_cond_path, self.work_dir)
        shutil.copy(melt_params_path, self.work_dir)
        shutil.copy(region_mask_path, self.work_dir)

        # Find and copy correct forcing files
        smb_search_path = os.path.join(forcing_basepath, exp_fcg, 'processed_SMB_*_smbNeg_over_bareland.nc')
        fcgFileList = glob.glob(smb_search_path)
        if len(fcgFileList) == 1:
            smb_path = fcgFileList[0]
            smb_fname = os.path.split(smb_path)[-1]
            shutil.copy(smb_path, self.work_dir)
        else:
            sys.exit(f"ERROR: Did not find exactly one matching SMB file at {smb_search_path}: {fcgFileList}")

        tf_search_path = os.path.join(forcing_basepath, exp_fcg, 'processed_TF_*.nc')
        fcgFileList = glob.glob(tf_search_path)
        if len(fcgFileList) == 1:
            tf_path = fcgFileList[0]
            tf_fname = os.path.split(tf_path)[-1]
            shutil.copy(tf_path, self.work_dir)
        else:
            sys.exit(f"ERROR: Did not find exactly one matching TF file at {tf_search_path}: {fcgFileList}")

        # Make stream modifications based on files that were determined above
        stream_replacements = {
                               'input_file_SMB_forcing': smb_fname,
                               'input_file_TF_forcing': tf_fname
                               }
        if self.exp == 'hist':
            stream_replacements['input_file_init_cond'] = init_cond_fname
            stream_replacements['input_file_region_mask'] = region_mask_fname
            stream_replacements['input_file_melt_params'] = melt_params_fname
        else:
            stream_replacements['input_file_init_cond'] = 'USE_RESTART_FILE_INSTEAD'
            stream_replacements['input_file_region_mask'] = 'USE_RESTART_FILE_INSTEAD'
            stream_replacements['input_file_melt_params'] = 'USE_RESTART_FILE_INSTEAD'
        if self.exp == 'hist' or self.exp == 'ctrlAE':
            stream_replacements['forcing_interval'] = 'initial_only'
        else:
            stream_replacements['forcing_interval'] = '0001-00-00_00:00:00'

        self.add_streams_file(
            'compass.landice.tests.ismip6_run_ais', 'streams.landice.template',
            out_name='streams.landice',
            template_replacements=stream_replacements)

        # Set up namelist and customize as needed
        self.add_namelist_file(
            'compass.landice.tests.ismip6_run_ais', 'namelist.landice',
            out_name='namelist.landice')

        if mesh_res == '04':
            options = {'config_pio_num_iotasks': '60',
                       'config_pio_stride': '64'}
            self.add_namelist_options(options=options,
                                      out_name='namelist.landice')

        if self.exp == 'hist':
            options = {'config_do_restart': ".false.",
                       'config_start_time': "'2000-01-01_00:00:00'",
                       'config_stop_time': "'2015-01-01_00:00:00'"}
            self.add_namelist_options(options=options,
                                      out_name='namelist.landice')

        # For all projection runs, symlink the restart file for the historical run
        # don't symlink restart_timestamp or you'll have a mighty mess
        if not self.exp == 'hist':
            os.symlink(f"../hist/rst.2015-01-01.nc", os.path.join(self.work_dir, 'rst.2015-01-01.nc'))
            with open(os.path.join(self.work_dir, "restart_timestamp"), "w") as text_file:
                text_file.write("2015-01-01_00:00:00")


        # add the albany_input.yaml file
        self.add_input_file(filename='albany_input.yaml',
                            package='compass.landice.tests.ismip6_run_ais',
                            copy=True)

        # copy graph files
        # may be possible to use compass functionality, but wasn't working in this way
        #make_graph_file(mesh_filename=init_cond_path,
        #                graph_filename='graph.info')
        graphFileList = glob.glob(os.path.join(graph_files_path, 'graph.info*'))
        for gf in graphFileList:
            shutil.copy(gf, self.work_dir)

        # provide an example submit script
        template = Template(resources.read_text(
            'compass.landice.tests.ismip6_run_ais',
            f'slurm.{mesh_res}.run'))
        slurm_replacements = {'EXP': f'{self.exp}_{mesh_res}',
                              'LOAD_SCRIPT': compass_load_path}
        rendered_text = template.render(slurm_replacements)
        with open(os.path.join(self.work_dir, 'slurm.run'), "w") as fh:
            fh.write(rendered_text)

        # link in exe
        self.add_model_as_input()

    def run(self):
        """
        Run this step of the test case
        """
        #make_graph_file(mesh_filename=self.mesh_file,
        #                graph_filename='graph.info')
        #run_model(step=self, namelist='namelist.landice',
        #          streams='streams.landice')
