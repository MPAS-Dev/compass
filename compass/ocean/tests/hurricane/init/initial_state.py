import json
import os
from importlib import resources

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
        else:
            self.add_streams_file(package, 'streams.init', mode='init')

        if not use_lts:

            mesh_path = mesh.steps['cull_mesh'].path

            self.add_input_file(
                filename='mesh.nc',
                work_dir_target=f'{mesh_path}/culled_mesh.nc')
            if self.wetdry != 'subgrid':
                # subgrid uses a specialized weighted graph file
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
            self.add_input_file(
                filename='graph.info.noweights',
                work_dir_target=f'{mesh_path}/culled_graph.info')

        self.add_model_as_input()

        if self.wetdry == 'subgrid':
            self.add_output_file(filename='ocean_subgrid_final.nc')
        else:
            self.add_output_file(filename='ocean.nc')
        self.add_output_file(filename='graph.info')

    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies
        """

        package = 'compass.ocean.tests.hurricane.init'
        if self.wetdry == 'subgrid':

            filename = 'bathy_data.json'
            with resources.open_text(package, filename) as bathy_file:
                self.bathy_files = json.load(bathy_file)

            self.add_namelist_file(package, 'namelist.init_subgrid',
                                   mode='init')
            dem = self.bathy_files["NCEI"][0]
            options = dict(
                config_subgrid_topography_file=f"'NCEI_data/{dem}'")
            self.add_namelist_options(
                options=options, mode='init',
                out_name='namelist.ocean')

            self.add_streams_file(package, 'streams.ocean_subgrid0',
                                  mode='init')

            os.makedirs(f'{self.work_dir}/NCEI_data', exist_ok=True)
            os.makedirs(f'{self.work_dir}/LULC_data', exist_ok=True)
            nfiles = len(self.bathy_files["NCEI"])
            for i, dem in enumerate(self.bathy_files["NCEI"]):
                self.add_input_file(
                    filename=f'NCEI_data/{dem}',
                    target=f'ncei/{dem}',
                    database='bathymetry_database')

                self.add_input_file(
                    filename=f'LULC_data/landuse_from_{dem}',
                    target=f'LULC/landuse_from_{dem}',
                    database='hurricane')

                options = dict(
                    config_subgrid_topography_file=f"'NCEI_data/{dem}'",
                    config_subgrid_lulc_file=f"'LULC_data/landuse_from_{dem}'")
                self.add_namelist_file(package, 'namelist.init', mode='init',
                                       out_name=f'namelist.ocean_subgrid{i}')
                self.add_namelist_file(package, 'namelist.init.wd',
                                       mode='init',
                                       out_name=f'namelist.ocean_subgrid{i}')
                self.add_namelist_file(package, 'namelist.init_subgrid',
                                       mode='init',
                                       out_name=f'namelist.ocean_subgrid{i}')
                self.add_namelist_options(
                    options=options, mode='init',
                    out_name=f'namelist.ocean_subgrid{i}')

                if i == nfiles - 1:
                    stream_replacements = {
                        'output_file': 'ocean_subgrid_final.nc',
                        'input_file': f'ocean_subgrid{i}.nc'}
                else:
                    stream_replacements = {
                        'output_file': f'ocean_subgrid{i + 1}.nc',
                        'input_file': f'ocean_subgrid{i}.nc'}
                if i == 0:
                    stream_replacements['input_file'] = 'ocean_init.nc'
                self.add_streams_file(
                    package, 'streams.template',
                    template_replacements=stream_replacements,
                    out_name=f'streams.ocean_subgrid{i}')

        self._get_resources()

    def constrain_resources(self, available_resources):
        """
        Update resources at runtime from config options
        """
        self._get_resources()
        super().constrain_resources(available_resources)

    def update_namelist_pio(self, out_name=None):
        """
        Modify the namelist so the number of PIO tasks and the stride between
        them consistent with the number of nodes and cores (one PIO task per
        node).

        Parameters
        ----------
        out_name : str, optional
            The name of the namelist file to write out, ``namelist.<core>`` by
            default
        """
        config = self.config
        cores = self.ntasks * self.cpus_per_task

        if out_name is None:
            out_name = f'namelist.{self.mpas_core.name}'

        cores_per_node = config.getint('parallel', 'cores_per_node')

        # update PIO tasks based on the machine settings and the available
        # number or cores
        pio_num_iotasks = 4 * int(np.ceil(cores / cores_per_node))
        pio_stride = self.ntasks // pio_num_iotasks
        if pio_stride > cores_per_node:
            raise ValueError(f'Not enough nodes for the number of cores.  '
                             f'cores: {cores}, cores per node: '
                             f'{cores_per_node}')

        replacements = {'config_pio_num_iotasks': f'{pio_num_iotasks}',
                        'config_pio_stride': f'{pio_stride}'}

        self.update_namelist_at_runtime(options=replacements,
                                        out_name=out_name)

    def run(self):
        """
        Run this step of the testcase
        """

        if self.wetdry == 'subgrid':
            self.load_balance_graphfile(min_lon=-190, max_lon=190,
                                        min_lat=-100, max_lat=100,
                                        inside_weight=1)
        run_model(self)

        if self.wetdry == 'subgrid':
            for i, dem in enumerate(self.bathy_files["NCEI"]):

                ds = nc.Dataset(f"NCEI_data/{dem}")
                lon = ds.variables["lon"][:]
                lat = ds.variables["lat"][:]
                min_lon = np.min(lon)
                max_lon = np.max(lon)
                min_lat = np.min(lat)
                max_lat = np.max(lat)

                self.load_balance_graphfile(min_lon, max_lon, min_lat, max_lat)
                run_model(self, namelist=f'namelist.ocean_subgrid{i}',
                          streams=f'streams.ocean_subgrid{i}')

                if os.path.isfile(f'ocean_subgrid{i - 2}.nc'):
                    os.remove(f'ocean_subgrid{i - 2}.nc')

        mesh = nc.Dataset("mesh.nc", "r")
        if self.wetdry == 'subgrid':
            init = nc.Dataset("ocean_subgrid_final.nc", "r+")
        else:
            init = nc.Dataset("ocean.nc", "r+")

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

        # lat_cutoff = -59.0
        lat_cutoff = 90.0
        lat_mask = np.where(init["latCell"][:] < lat_cutoff * np.pi / 180.0)[0]
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

    def load_balance_graphfile(self, min_lon, max_lon,
                               min_lat, max_lat, inside_weight=1000):
        weights = []

        outside_weight = 1

        mesh_filename = 'mesh.nc'
        graph_filename = "graph.info.noweights"
        output_filename = "graph.info"

        # Read the cell centers from the CSV file.
        grid_nc = nc.Dataset(mesh_filename, 'r')
        lon_grid = grid_nc.variables['lonCell'][:] * 180.0 / np.pi
        lat_grid = grid_nc.variables['latCell'][:] * 180.0 / np.pi
        lon_grid = np.mod(lon_grid + 180.0, 360.0) - 180.0
        nCells = lon_grid.size

        inside_count = 0
        for iCell in range(nCells):
            lon = lon_grid[iCell]
            lat = lat_grid[iCell]
            if self.is_inside(lon, lat, min_lon, max_lon, min_lat, max_lat):
                weights.append(inside_weight)
                inside_count = inside_count + 1
            else:
                weights.append(outside_weight)
        print(f'cells inside: {inside_count}/{nCells}')

        f = open(graph_filename, 'r')
        lines = f.read().splitlines()
        weight_lines = []
        for i, line in enumerate(lines):
            if i != 0:
                weight_lines.append(f'{weights[i - 1]} {line}')
            else:
                weight_lines.append(f'{line} 010')

        # Write the weights to the output file, one weight per line.
        with open(output_filename, "w") as outfile:
            for line in weight_lines:
                outfile.write(f"{line}\n")

    def is_inside(self, lon, lat, min_lon, max_lon, min_lat, max_lat):
        """
        Determine if a given coordinate (lon, lat) is within the bounding box.
        """
        return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)
