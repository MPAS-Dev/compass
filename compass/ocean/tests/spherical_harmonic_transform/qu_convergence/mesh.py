import numpy as np

from mpas_tools.ocean import build_spherical_mesh
from mpas_tools.logging import check_call

from compass.step import Step
from compass.model import make_graph_file

import os


class Mesh(Step):
    """
    A step for creating uniform global meshes

    Attributes
    ----------
    resolution : int
        The resolution of the (uniform) mesh in km

    serial_nLat : int
        The number of latitudes in the Gaussian grid
    """
    def __init__(self, test_case, resolution, serial_nLat):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.spherical_harmonic_transform.qu_convergence.QuConvergence
            The test case this step belongs to

        resolution : int
            The resolution of the (uniform) mesh in km

        serial_nLat : int
            The number of latitudes in the Gaussian grid

        """
        super().__init__(test_case=test_case,
                         name=f'QU{resolution}_mesh',
                         subdir=f'QU{resolution}/mesh',
                         cpus_per_task=18, min_cpus_per_task=1,
                         openmp_threads=1)
        for file in ['mesh.nc', 'graph.info']:
            self.add_output_file(filename=file)

        self.resolution = resolution
        self.serial_nLat = serial_nLat

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger

        # only use progress bars if we're not writing to a log file
        use_progress_bar = self.log_filename is None

        # create the base mesh
        cellWidth, lon, lat = self.build_cell_width_lat_lon()
        build_spherical_mesh(cellWidth, lon, lat, out_filename='mesh.nc',
                             logger=logger, use_progress_bar=use_progress_bar)

        make_graph_file(mesh_filename='mesh.nc',
                        graph_filename='graph.info')

        for nLat in self.serial_nLat:
            self.mapping_files(nLat)

    def build_cell_width_lat_lon(self):
        """
        Create cell width array for this mesh on a regular latitude-longitude
        grid
        Returns
        -------
        cellWidth : numpy.array
            m x n array of cell width in km

        lon : numpy.array
            longitude in degrees (length n and between -180 and 180)

        lat : numpy.array
            longitude in degrees (length m and between -90 and 90)
        """
        dlon = 10.
        dlat = dlon
        constantCellWidth = float(self.resolution)

        nlat = int(180/dlat) + 1
        nlon = int(360/dlon) + 1
        lat = np.linspace(-90., 90., nlat)
        lon = np.linspace(-180., 180., nlon)

        cellWidth = constantCellWidth * np.ones((lat.size, lon.size))
        return cellWidth, lon, lat

    def mapping_files(self, N):

        nLat = str(N)
        nLon = 2*int(N)
        nLon = str(nLon)

        parallel_executable = self.config.get('parallel',
                                              'parallel_executable')
        ntasks = self.ntasks
        parallel_args = parallel_executable.split(' ')
        if 'srun' in parallel_args:
            parallel_args.extend(['-n', f'{ntasks}'])
        else:  # presume mpirun syntax
            parallel_args.extend(['-np', f'{ntasks}'])

        if not os.path.exists('mpas_mesh_scrip.nc'):
            args = ['scrip_from_mpas',
                    '-m', 'mesh.nc',
                    '-s', 'mpas_mesh_scrip.nc']
            check_call(args, self.logger)

        if not os.path.exists('mpas_to_grid_'+nLat+'.nc'):
            gauss_args = f'ttl={nLat}x{nLon}' \
                         f'#latlon={nLat},{nLon}' \
                          '#lat_typ=gss' \
                          '#lon_typ=grn_ctr'
            args = ['ncremap',
                    '-G', gauss_args,
                    '-g', 'gaussian_grid_scrip.nc']
            check_call(args, self.logger)

            args = ['ESMF_RegridWeightGen',
                    '-d',  'gaussian_grid_scrip.nc',
                    '-s',  'mpas_mesh_scrip.nc',
                    '-w', f'mpas_to_grid_{nLat}.nc',
                    '-i']
            args = parallel_args + args
            check_call(args, self.logger)

            args = ['ESMF_RegridWeightGen',
                    '-s',  'gaussian_grid_scrip.nc',
                    '-d',  'mpas_mesh_scrip.nc',
                    '-w', f'grid_to_mpas_{nLat}.nc',
                    '-i']
            args = parallel_args + args
            check_call(args, self.logger)
