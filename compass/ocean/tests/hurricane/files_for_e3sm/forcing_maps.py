from mpas_tools.logging import check_call
from pyremap import MpasCellMeshDescriptor

from compass.parallel import run_command
from compass.step import Step


class ForcingMaps(Step):
    """
    A step for building mapping files for remapping forcing files
    to a global MPAS-Ocean mesh

    Attributes
    ----------
    mesh : compass.mesh.spherical.SphericalBaseStep
        The base mesh step containing input files to this step
    """

    def __init__(self, test_case):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.hurricane.files_for_e3sm.FilesForE3SM
            The test case this step belongs to
        """
        super().__init__(test_case, name='forcing_maps')
        self.mesh = test_case.mesh
        self.creation_date = test_case.creation_date

    def setup(self):
        """
        Set up the step in the work directory, including downloading any
        dependencies.
        """
        super().setup()

        atm_grid = self.config.get('files_for_e3sm', 'atm_grid')
        ocn_grid = self.mesh.mesh_name
        mesh_path = self.mesh.steps['cull_mesh'].path
        create_date = self.creation_date

        self.add_input_file(
            filename='mesh.nc', work_dir_target=f'{mesh_path}/culled_mesh.nc')
        self.add_output_file(
            filename=f'map_{atm_grid}_to_{ocn_grid}_trbilin.{create_date}.nc')
        self.add_output_file(
            filename=f'map_{atm_grid}_to_{ocn_grid}_traave.{create_date}.nc')
        self.add_output_file(
            filename=f'map_{ocn_grid}_to_{atm_grid}_trbilin.{create_date}.nc')
        self.add_output_file(
            filename=f'map_{ocn_grid}_to_{atm_grid}_traave.{create_date}.nc')

        self._get_resources()

    def constrain_resources(self, available_resources):
        """
        Constrain ``cpus_per_task`` and ``ntasks`` based on the number of
        cores available to this step

        Parameters
        ----------
        available_resources : dict
            The total number of cores available to the step
        """
        self._get_resources()
        super().constrain_resources(available_resources)

    def run(self):
        """
        Run this step of the test case
        """
        super().run()

        atm_grid = self.config.get('files_for_e3sm', 'atm_grid')
        ocn_grid = self.mesh.mesh_name

        self._scrip_file_gridded()
        self._scrip_file_MPAS()
        self._partition_scrip_file(atm_grid)
        self._partition_scrip_file(ocn_grid)
        self._create_weights(atm_grid, ocn_grid, 'trbilin')
        self._create_weights(atm_grid, ocn_grid, 'traave')
        self._create_weights(ocn_grid, atm_grid, 'trbilin')
        self._create_weights(ocn_grid, atm_grid, 'traave')

    def _get_resources(self):
        """
        Get resources
        """
        section = self.config['hurricane']
        self.ntasks = section.getint('init_ntasks')
        self.min_tasks = section.getint('init_min_tasks')
        self.openmp_threads = section.getint('init_threads')

    def _scrip_file_gridded(self):
        """
        Create gridded SCRIP file for atm
        """
        logger = self.logger
        grid_name = self.config.get('files_for_e3sm', 'atm_grid')
        logger.info(f'Create gridded SCRIP file for {grid_name} grid')

        grids = {
            'T382': {
                'nlat': 576,
                'nlon': 1152,
                'lat_typ': 'gss',
                'lon_typ': 'grn_ctr',
            },
            'T574': {
                'nlat': 880,
                'nlon': 1760,
                'lat_typ': 'gss',
                'lon_typ': 'grn_ctr',
            },
        }

        nlat, nlon, lat_typ, lon_typ = grids[grid_name].values()
        args = [
            'ncremap',
            '-G', f'latlon={nlat},{nlon}#lat_typ={lat_typ}#lon_typ={lon_typ}',
            '-g', f'{grid_name}.scrip.nc',
        ]
        check_call(args, logger)

        logger.info('  Done.')

    def _scrip_file_MPAS(self):
        """
        Create SCRIP file from MPAS mesh file.
        """
        grid_name = self.mesh.mesh_name
        logger = self.logger
        logger.info(f'Create MPAS SCRIP file for {grid_name} mesh')

        descriptor = MpasCellMeshDescriptor(
            filename='mesh.nc',
            mesh_name=grid_name,
        )
        descriptor.to_scrip(f'{grid_name}.scrip.nc')

        logger.info('  Done.')

    def _partition_scrip_file(self, grid_name):
        """
        Partition SCRIP file for parallel mbtempest use
        """
        logger = self.logger
        logger.info(f'Partition SCRIP file for {grid_name}')

        # Convert source SCRIP to mbtempest
        args = [
            'mbconvert', '-B',
            f'{grid_name}.scrip.nc',
            f'{grid_name}.scrip.h5m',
        ]
        # run in "parallel" with one task and one thread for Intel-MPI support
        run_command(args, 1, 1, 1, self.config, logger)

        # Partition source SCRIP
        args = [
            'mbpart', f'{self.ntasks}',
            '-z', 'RCB',
            f'{grid_name}.scrip.h5m',
            f'{grid_name}.scrip.p{self.ntasks}.h5m',
        ]
        # run in "parallel" with one task and one thread for Intel-MPI support
        run_command(args, 1, 1, 1, self.config, logger)

        logger.info('  Done.')

    def _create_weights(self, src, tgt, method):
        """
        Create mapping weights file using TempestRemap
        """
        logger = self.logger
        logger.info('Create weights file')

        if method not in ['traave', 'trbilin']:
            raise ValueError(f'Unsupported regridding method {method}')

        src_file = f'{src}.scrip.p{self.ntasks}.h5m'
        tgt_file = f'{tgt}.scrip.p{self.ntasks}.h5m'
        map_file = f'map_{src}_to_{tgt}_{method}.{self.creation_date}.nc'

        # Build weights file
        args = [
            'mbtempest', '--type', '5', '--weights',
            '--load', src_file,
            '--load', tgt_file,
            '--file', map_file,
            '--method', 'fv', '--order', '1',
            '--method', 'fv', '--order', '1',
        ]
        if method == 'trbilin':
            args.extend(['--fvmethod', 'bilin'])
        run_command(
            args, self.cpus_per_task, self.ntasks,
            self.openmp_threads, self.config, self.logger,
        )

        # Add attributes required by `generate_domain_files_E3SM.py`
        args = [
            'ncatted', '-O',
            '-a', f'grid_file_src,global,o,c,{src_file}',
            '-a', f'grid_file_dst,global,o,c,{tgt_file}',
            map_file,
        ]
        check_call(args, logger)

        logger.info('  Done.')
