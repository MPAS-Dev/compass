import netCDF4 as nc
import numpy as np

from compass.model import run_model
from compass.ocean.plot import plot_vertical_grid
from compass.ocean.vertical.grid_1d import generate_1d_grid, write_1d_grid
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for tidal
    test cases

    Attributes
    ----------
    mesh : compass.ocean.tests.tides.mesh.mesh.MeshStep
        The step for creating the mesh

    """
    def __init__(self, test_case, mesh):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.tides.init.Init
            The test case this step belongs to

        mesh : compass.ocean.tests.tides.mesh.Mesh
            The test case that creates the mesh used by this test case

        """

        super().__init__(test_case=test_case, name='initial_state')
        self.mesh = mesh

        package = 'compass.ocean.tests.tides.init'

        # generate the namelist, replacing a few default options
        self.add_namelist_file(package, 'namelist.init', mode='init')

        # generate the streams file
        self.add_streams_file(package, 'streams.init', mode='init')

        mesh_path = mesh.steps['cull_mesh'].path

        self.add_input_file(
            filename='mesh.nc',
            work_dir_target=f'{mesh_path}/culled_mesh.nc')

        self.add_input_file(
            filename='graph.info',
            work_dir_target=f'{mesh_path}/culled_graph.info')

        self.add_input_file(
            target='PotentialTemperature.01.filled.60levels.PHC.151106.nc',
            filename='temperature.nc',
            database='initial_condition_database')

        self.add_input_file(
            target='Salinity.01.filled.60levels.PHC.151106.nc',
            filename='salinity.nc',
            database='initial_condition_database')

        self.add_input_file(
            target='windStress.ncep_1958-2000avg.interp3600x2431.151106.nc',
            filename='wind_stress.nc',
            database='initial_condition_database')

        self.add_input_file(
            target='chlorophyllA_monthly_averages_1deg.151201.nc',
            filename='swData.nc',
            database='initial_condition_database')

        self.add_input_file(
            target='BedMachineAntarctica_and_GEBCO_2019_0.05_degree.200128.nc',
            filename='topography.nc',
            database='bathymetry_database')

        self.add_model_as_input()

        for file in ['initial_state.nc', 'graph.info']:
            self.add_output_file(filename=file)

    def setup(self):
        """
        Get resources at setup from config options
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

        config = self.config
        interfaces = generate_1d_grid(config=config)

        write_1d_grid(interfaces=interfaces, out_filename='vertical_grid.nc')
        plot_vertical_grid(grid_filename='vertical_grid.nc', config=config,
                           out_filename='vertical_grid.png')

        run_model(self)

        max_depth = config.getfloat('vertical_grid', 'bottom_depth')
        min_depth = config.getfloat('vertical_grid', 'min_depth')

        init = nc.Dataset("initial_state.nc", "r+")
        init["bottomDepth"][:] = \
            np.minimum(max_depth,
                       np.maximum(min_depth, init["bottomDepthObserved"][:]))

        init["layerThickness"][0, :, 0] = init["bottomDepth"][:]
        init["restingThickness"][:, 0] = init["bottomDepth"][:]

        # -- Estimate vert. grid for ice-shelves, min.-thicknesses, etc
        # -- Darren Engwirda

        print("Est. layering to account for ice-shelves")

        mesh = nc.Dataset("mesh.nc", "r")

        botd = np.asarray(init["bottomDepthObserved"][:], dtype=np.float64)
        # ossh = np.asarray(init["ssh"][0,:], dtype=np.float64)
        ossh = 0. * botd  # assume ssh is zero

        grav = 9.80665  # gravitational accel.
        irho = float(init.config_land_ice_flux_rho_ice)
        orho = float(init.config_density0)
        minh = float(init.config_drying_min_cell_height) / 2.

        print("ice-shelf density:", irho)
        print("ocn-const density:", orho)
        print("min-layer thickness:", minh)

        iceh = np.asarray(mesh["ice_thickness"][:], dtype=np.float64)
        # icef = np.asarray(mesh["ice_cover"][:], dtype=np.float64)

        icep = irho * grav * iceh  # ice pressure
        iced = icep / grav / orho  # ice draft

        # ensure thin-layer beneath ice-shelves
        iced = np.minimum(iced, +botd - minh)
        iced = np.maximum(iced, +0.0)
        ossh = ossh - iced

        icep[iced <= 0.] = 0.

        # allow thin-layer in partially flooded zone
        ossh = np.maximum(ossh, -botd + minh)

        print("max ice-draft:", np.max(iced))
        print("max ice-pressure:", np.max(icep))

        if ("ssh" not in init.variables.keys()):
            init.createVariable("ssh", "f8", ("Time", "nCells"))

        if ("landIceDraft" not in init.variables.keys()):
            init.createVariable("landIceDraft", "f8", ("Time", "nCells"))

        if ("landIcePressure" not in init.variables.keys()):
            init.createVariable("landIcePressure", "f8", ("Time", "nCells"))

        if ("landIceMask" not in init.variables.keys()):
            init.createVariable("landIceMask", "i4", ("Time", "nCells"))

        if ("landIceFloatingMask" not in init.variables.keys()):
            init.createVariable("landIceFloatingMask", "i4",
                                ("Time", "nCells"))

        if ("landIceFraction" not in init.variables.keys()):
            init.createVariable("landIceFraction", "f8", ("Time", "nCells"))

        if ("landIceFloatingFraction" not in init.variables.keys()):
            init.createVariable("landIceFloatingFraction", "f8",
                                ("Time", "nCells"))

        init["landIceDraft"][0, :] = -iced  # NB. sign
        init["landIcePressure"][0, :] = icep
        init["landIceMask"][0, :] = (icep > 0.)
        init["landIceFloatingMask"][0, :] = (icep > 0.)
        init["landIceFraction"][0, :] = (icep > 0.)
        init["landIceFloatingFraction"][0, :] = (icep > 0.)

        init["bottomDepth"][:] = botd
        init["ssh"][0, :] = ossh

        init["layerThickness"][0, :, 0] = ossh + botd
        # init["restingThickness"][:,0] = ossh + botd

        init.close()

    def _get_resources(self):
        # get the these properties from the config options
        config = self.config
        self.ntasks = config.getint('tides', 'init_ntasks')
        self.min_tasks = config.getint('tides', 'init_min_tasks')
        self.openmp_threads = config.getint('tides', 'init_threads')
