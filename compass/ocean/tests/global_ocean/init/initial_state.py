from compass.ocean.tests.global_ocean.metadata import \
    add_mesh_and_init_metadata
from compass.model import run_model
from compass.ocean.vertical.grid_1d import generate_1d_grid, write_1d_grid
from compass.ocean.plot import plot_vertical_grid, plot_initial_state
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for baroclinic channel
    test cases

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.mesh.MeshStep
        The step for creating the mesh

    initial_condition : {'PHC', 'EN4_1900'}
        The initial condition dataset to use

    with_bgc : bool
        Whether to include biogeochemistry (BGC) in the initial condition
    """
    def __init__(self, test_case, mesh, initial_condition, with_bgc):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.init.Init
            The test case this step belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that creates the mesh used by this test case

        initial_condition : {'PHC', 'EN4_1900'}
            The initial condition dataset to use

        with_bgc : bool
            Whether to include biogeochemistry (BGC) in the initial condition
        """
        if initial_condition not in ['PHC', 'EN4_1900']:
            raise ValueError(f'Unknown initial_condition {initial_condition}')

        super().__init__(test_case=test_case, name='initial_state')
        self.mesh = mesh
        self.initial_condition = initial_condition
        self.with_bgc = with_bgc

        package = 'compass.ocean.tests.global_ocean.init'

        # generate the namelist, replacing a few default options
        self.add_namelist_file(package, 'namelist.init', mode='init')
        self.add_namelist_file(
            package, f'namelist.{initial_condition.lower()}',
            mode='init')
        if mesh.with_ice_shelf_cavities:
            self.add_namelist_file(package, 'namelist.wisc', mode='init')
        if with_bgc:
            self.add_namelist_file(package, 'namelist.bgc', mode='init')

        # generate the streams file
        self.add_streams_file(package, 'streams.init', mode='init')

        if mesh.with_ice_shelf_cavities:
            self.add_streams_file(package, 'streams.wisc', mode='init')

        self.add_input_file(
            filename='topography.nc',
            target='BedMachineAntarctica_and_GEBCO_2019_0.05_degree.200128.nc',
            database='bathymetry_database')

        self.add_input_file(
            filename='wind_stress.nc',
            target='windStress.ncep_1958-2000avg.interp3600x2431.151106.nc',
            database='initial_condition_database')

        self.add_input_file(
            filename='swData.nc',
            target='chlorophyllA_monthly_averages_1deg.151201.nc',
            database='initial_condition_database')

        if initial_condition == 'PHC':
            self.add_input_file(
                filename='temperature.nc',
                target='PotentialTemperature.01.filled.60levels.PHC.151106.nc',
                database='initial_condition_database')
            self.add_input_file(
                filename='salinity.nc',
                target='Salinity.01.filled.60levels.PHC.151106.nc',
                database='initial_condition_database')
        else:
            # EN4_1900
            self.add_input_file(
                filename='temperature.nc',
                target='PotentialTemperature.100levels.Levitus.'
                       'EN4_1900estimate.200813.nc',
                database='initial_condition_database')
            self.add_input_file(
                filename='salinity.nc',
                target='Salinity.100levels.Levitus.EN4_1900estimate.200813.nc',
                database='initial_condition_database')

        if with_bgc:
            self.add_input_file(
                filename='ecosys.nc',
                target='ecosys_jan_IC_360x180x60_corrO2_Dec2014phaeo.nc',
                database='initial_condition_database')
            self.add_input_file(
                filename='ecosys_forcing.nc',
                target='ecoForcingAllSurface.forMPASO.interp360x180.'
                       '1timeLevel.nc',
                database='initial_condition_database')

        mesh_path = mesh.mesh_step.path

        self.add_input_file(
            filename='mesh.nc',
            work_dir_target=f'{mesh_path}/culled_mesh.nc')

        self.add_input_file(
            filename='graph.info',
            work_dir_target=f'{mesh_path}/culled_graph.info')

        if mesh.with_ice_shelf_cavities:
            self.add_input_file(
                filename='land_ice_mask.nc',
                work_dir_target=f'{mesh_path}/land_ice_mask.nc')

        self.add_model_as_input()

        for file in ['initial_state.nc', 'init_mode_forcing_data.nc',
                     'graph.info']:
            self.add_output_file(filename=file)

    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies.
        """
        # get the these properties from the config options
        config = self.config
        self.ntasks = config.getint('global_ocean', 'init_ntasks')
        self.min_tasks = config.getint('global_ocean', 'init_min_tasks')
        self.openmp_threads = config.getint('global_ocean', 'init_threads')

    def run(self):
        """
        Run this step of the testcase
        """
        config = self.config
        interfaces = generate_1d_grid(config=config)

        write_1d_grid(interfaces=interfaces, out_filename='vertical_grid.nc')
        plot_vertical_grid(grid_filename='vertical_grid.nc', config=config,
                           out_filename='vertical_grid.png')

        run_model(self)

        add_mesh_and_init_metadata(self.outputs, config,
                                   init_filename='initial_state.nc')

        plot_initial_state(input_file_name='initial_state.nc',
                           output_file_name='initial_state.png')
