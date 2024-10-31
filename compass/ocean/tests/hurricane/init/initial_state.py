from compass.model import run_model
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for hurricane
    test cases

    Attributes
    ----------
    mesh : compass.ocean.tests.hurricane.mesh.mesh.MeshStep
        The step for creating the mesh

    """
    def __init__(self, test_case, mesh, use_lts, wetdry):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.hurricane.init.Init
            The test case this step belongs to

        mesh : compass.ocean.tests.hurricane.mesh.Mesh
            The test case that creates the mesh used by this test case

        use_lts: bool
            Whether local time-stepping is used

        """

        super().__init__(test_case=test_case, name='initial_state')
        self.mesh = mesh
        self.use_lts = use_lts
        self.wetdry = wetdry

        package = 'compass.ocean.tests.hurricane.init'

        # generate the namelist, replacing a few default options
        self.add_namelist_file(package, 'namelist.init', mode='init')

        if mesh.mesh_name == 'DEVR45to5rr1':
            self.add_namelist_file(package, 'namelist.init.wd', mode='init')

        # generate the streams file
        if self.wetdry == 'subgrid':
            self.add_namelist_file(package, 'namelist.init_subgrid',
                                   mode='init')
            self.add_streams_file(package, 'streams.ocean_subgrid0',
                                  mode='init')
            options = dict(
                config_Buttermilk_bay_topography_file="'crm_vol1_2023_nc3.nc'")
            self.add_namelist_options(options=options, mode='init',
                                      out_name='namelist.ocean')

            self.add_streams_file(package, 'streams.ocean_subgrid1',
                                  mode='init',
                                  out_name='streams.ocean_subgrid1')
            self.add_namelist_file(package, 'namelist.init', mode='init',
                                   out_name='namelist.ocean_subgrid1')
            self.add_namelist_file(package, 'namelist.init.wd', mode='init',
                                   out_name='namelist.ocean_subgrid1')
            self.add_namelist_file(package, 'namelist.init_subgrid',
                                   mode='init',
                                   out_name='namelist.ocean_subgrid1')
            self.add_namelist_options(options=options, mode='init',
                                      out_name='namelist.ocean_subgrid1')

            self.add_streams_file(package, 'streams.ocean_subgrid2',
                                  mode='init',
                                  out_name='streams.ocean_subgrid2')
            self.add_namelist_file(package, 'namelist.init', mode='init',
                                   out_name='namelist.ocean_subgrid2')
            self.add_namelist_file(package, 'namelist.init.wd', mode='init',
                                   out_name='namelist.ocean_subgrid2')
            self.add_namelist_file(package, 'namelist.init_subgrid',
                                   mode='init',
                                   out_name='namelist.ocean_subgrid2')
            options = dict(
                config_Buttermilk_bay_topography_file="'crm_vol2_2023_nc3.nc'")
            self.add_namelist_options(options=options, mode='init',
                                      out_name='namelist.ocean_subgrid2')
        else:
            self.add_streams_file(package, 'streams.init', mode='init')

        if not use_lts:

            mesh_path = mesh.steps['cull_mesh'].path

            self.add_input_file(
                filename='mesh.nc',
                work_dir_target=f'{mesh_path}/culled_mesh.nc')

            self.add_input_file(
                filename='graph.info',
                work_dir_target=f'{mesh_path}/culled_graph.info')

        else:

            mesh_path = mesh.steps['lts_regions'].path

            self.add_input_file(
                filename='mesh.nc',
                work_dir_target=f'{mesh_path}/lts_mesh.nc')

            self.add_input_file(
                filename='graph.info',
                work_dir_target=f'{mesh_path}/lts_graph.info')

        if self.wetdry == 'subgrid':
            self.add_input_file(filename='crm_vol1_2023_nc3.nc',
                                target='crm_vol1_2023_nc3.nc',
                                database='bathymetry_database')

            self.add_input_file(filename='crm_vol2_2023_nc3.nc',
                                target='crm_vol2_2023_nc3.nc',
                                database='bathymetry_database')

        self.add_model_as_input()

        if self.wetdry == 'subgrid':
            self.add_output_file(filename='ocean_subgrid2.nc')
        else:
            self.add_output_file(filename='ocean.nc')
        self.add_output_file(filename='graph.info')

    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies
        """
        self._get_resources()

    def constrain_resources(self, available_resources):
        """
        Update resources at runtime from config options
        """
        self._get_resources()
        super().constrain_resources(available_resources)

    def run(self):
        """
        Run this step of the testcase
        """
        run_model(self)
        if self.wetdry == 'subgrid':
            run_model(self, namelist='namelist.ocean_subgrid1',
                      streams='streams.ocean_subgrid1')
            run_model(self, namelist='namelist.ocean_subgrid2',
                      streams='streams.ocean_subgrid2')

    def _get_resources(self):
        # get the these properties from the config options
        config = self.config
        self.ntasks = config.getint('hurricane', 'init_ntasks')
        self.min_tasks = config.getint('hurricane', 'init_min_tasks')
        self.openmp_threads = config.getint('hurricane', 'init_threads')
