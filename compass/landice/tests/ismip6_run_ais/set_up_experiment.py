import os
from compass.model import run_model, make_graph_file
from compass.step import Step
import shutil, glob, sys


class SetUpExperiment(Step):
    """
    A step for setting up an ISMIP6 experiment

    Attributes
    ----------
    mesh_file : str
        The name of the mesh file being used

    mesh_type : str
        The resolution or mesh type of the test case
    """

    def __init__(self, test_case, name, subdir, exp):
        """
        Set up a new experiment

        Parameters
        ----------
        mesh_file : str
            The name of the mesh file being used

        mesh_type : {'high', 'mid'}
            The resolution or mesh type of the test case

        exp : experiment
        """

        self.exp = exp

        super().__init__(test_case=test_case, name=name)

    def setup(self):
        config = self.config
        section = config['ismip6_run_ais']
        forcing_basepath = section.get('forcing_basepath')
        init_cond_path = section.get('init_cond_path')
        init_cond_fname = os.path.split(init_cond_path)[-1]
        melt_params_path = section.get('melt_params_path')
        melt_params_fname = os.path.split(melt_params_path)[-1]
        region_mask_path = section.get('region_mask_path')
        region_mask_fname = os.path.split(region_mask_path)[-1]

        # Copy files we'll need from local paths specified in cfg file
        shutil.copy(init_cond_path, self.work_dir)
        shutil.copy(melt_params_path, self.work_dir)
        shutil.copy(region_mask_path, self.work_dir)

        # Find and copy correct forcing files
        smb_search_path = os.path.join(forcing_basepath, self.exp, 'processed_SMB_*_smbNeg_over_bareland.nc')
        fcgFileList = glob.glob(smb_search_path)
        if len(fcgFileList) == 1:
            smb_path = fcgFileList[0]
            smb_fname = os.path.split(smb_path)[-1]
            shutil.copy(smb_path, self.work_dir)
        else:
            sys.exit(f"ERROR: Did not find exactly one matching SMB file at {smb_search_path}: {fcgFileList}")

        tf_search_path = os.path.join(forcing_basepath, self.exp, 'processed_TF_*.nc')
        fcgFileList = glob.glob(tf_search_path)
        if len(fcgFileList) == 1:
            tf_path = fcgFileList[0]
            tf_fname = os.path.split(tf_path)[-1]
            shutil.copy(tf_path, self.work_dir)
        else:
            sys.exit(f"ERROR: Did not find exactly one matching TF file at {tf_search_path}: {fcgFileList}")

        # Make stream modifications based on files that were determined above
        stream_replacements = {
                               'input_file_init_cond': init_cond_fname ,
                               'input_file_region_mask': region_mask_fname ,
                               'input_file_melt_params': melt_params_fname,
                               'input_file_SMB_forcing': smb_fname,
                               'input_file_TF_forcing': tf_fname
                               }
        self.add_streams_file(
            'compass.landice.tests.ismip6_run_ais', 'streams.landice.template',
            out_name='streams.landice',
            template_replacements=stream_replacements)

        # Set up namelist and customize as needed
        self.add_namelist_file(
            'compass.landice.tests.ismip6_run_ais', 'namelist.landice',
            out_name='namelist.landice')
        #options = {'config_velocity_solver': f"'{velo_solver}'",
        #           'config_calving': f"'{calving_law}'"}

        # now add accumulated options to namelist
        #self.add_namelist_options(options=options,
        #                          out_name='namelist.landice')


        self.add_input_file(filename='albany_input.yaml',
                            package='compass.landice.tests.ismip6_run_ais',
                            copy=True)

        #make_graph_file(mesh_filename=self.mesh_file,
        #                graph_filename='graph.info')

        self.add_model_as_input()

    def run(self):
        """
        Run this step of the test case
        """
        #make_graph_file(mesh_filename=self.mesh_file,
        #                graph_filename='graph.info')
        #run_model(step=self, namelist='namelist.landice',
        #          streams='streams.landice')
