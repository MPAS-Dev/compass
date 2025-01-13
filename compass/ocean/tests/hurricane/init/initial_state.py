import netCDF4 as nc
import numpy as np

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

        init = nc.Dataset("ocean_subgrid2.nc", "r+")
        mesh = nc.Dataset("mesh.nc", "r")

        # -- Estimate vert. grid for ice-shelves, min.-thicknesses, etc
        # -- Darren Engwirda

        print("Est. layering to account for ice-shelves")

        botd = np.asarray(-mesh["bottomDepthObserved"][:], dtype=np.float64)
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

        zeros = np.zeros_like(iced)
        init["landIceDraft"][0, :] = zeros
        init["landIcePressure"][0, :] = zeros
        init["landIceMask"][0, :] = zeros
        init["landIceFloatingMask"][0, :] = zeros
        init["landIceFraction"][0, :] = zeros
        init["landIceFloatingFraction"][0, :] = zeros

        lat_mask = np.where(init["latCell"][:] < -59.0 * np.pi / 180.0)[0]
        init["landIceDraft"][0, lat_mask] = -iced[lat_mask]  # NB. sign
        init["landIcePressure"][0, lat_mask] = icep[lat_mask]
        init["landIceMask"][0, lat_mask] = (icep[lat_mask] > 0.)
        init["landIceFloatingMask"][0, lat_mask] = (icep[lat_mask] > 0.)
        init["landIceFraction"][0, lat_mask] = (icep[lat_mask] > 0.)
        init["landIceFloatingFraction"][0, lat_mask] = (icep[lat_mask] > 0.)

        init["bottomDepth"][lat_mask] = botd[lat_mask]
        ssh = init["ssh"][:]
        ssh[0, lat_mask] = ossh[lat_mask]
        init["ssh"][:] = ssh

        layerThickness = init["layerThickness"][:]
        layerThickness[0, lat_mask, 0] = ossh[lat_mask] + botd[lat_mask]
        init["layerThickness"][:] = layerThickness

        init.close()

    def _get_resources(self):
        # get the these properties from the config options
        config = self.config
        self.ntasks = config.getint('hurricane', 'init_ntasks')
        self.min_tasks = config.getint('hurricane', 'init_min_tasks')
        self.openmp_threads = config.getint('hurricane', 'init_threads')
