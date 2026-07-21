import shutil

from mpas_tools.logging import check_call
from mpas_tools.scrip.from_mpas import scrip_from_mpas

from compass.step import Step


class CreateSlmMappingFiles(Step):
    """
    A step for creating mapping files for the Sea Level Model
    """

    def __init__(self, test_case, name, subdir):
        """
        Initialize step
        """
        super().__init__(test_case=test_case, name=name, subdir=subdir)

    def setup(self):
        print("    Setting up mapping_file subdirectory")

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger
        section = config['ismip7_run_ais']
        sea_level_model = section.getboolean('sea_level_model')
        if sea_level_model:
            self._build_mapping_files(config, logger)

    def _build_mapping_files(self, config, logger):
        """
        Build mapping files between the MALI mesh and the SLM grid.
        """
        section = config['ismip7_run_ais']
        init_cond_path = section.get('init_cond_path')
        nglv = section.getint('nglv')
        section = config['parallel']
        ntasks = section.getint('cores_per_node')

        mali_scripfile = 'mali_scripfile.nc'
        slm_scripfile = f'slm_nglv{nglv}scripfile.nc'
        mali_meshfile = 'mali_meshfile_sphereLatLon.nc'

        # SLM scripfile
        logger.info(f'creating scripfile for the SLM grid with '
                    f'{nglv} Gauss-Legendre points in latitude')

        args = ['ncremap',
                '-g', slm_scripfile,
                '-G',
                f'latlon={nglv},{2 * int(nglv)}#lat_typ=gss#lat_drc=n2s']

        check_call(args, logger=logger)

        # MALI scripfile
        shutil.copy(init_cond_path, mali_meshfile)
        args = ['set_lat_lon_fields_in_planar_grid',
                '--file', mali_meshfile,
                '--proj', 'ais-bedmap2-sphere']

        check_call(args, logger=logger)

        logger.info('creating scrip file for the mali mesh')
        scrip_from_mpas(mali_meshfile, mali_scripfile)

        # MALI -> SLM mapping file
        logger.info('creating MALI -> SLM grid mapfile with conserve method')

        parallel_executable = config.get("parallel", "parallel_executable")
        args = parallel_executable.split(' ')
        args.extend(['-n', f'{ntasks}',
                     'ESMF_RegridWeightGen',
                     '-s', mali_scripfile,
                     '-d', slm_scripfile,
                     '-w', 'mapfile_mali_to_slm.nc',
                     '-m', 'conserve',
                     '-i', '-64bit_offset', '--netcdf4',
                     '--src_regional'])

        check_call(args, logger)

        # SLM -> MALI mapping file
        logger.info('creating SLM -> MALI mesh mapfile with bilinear method')
        args = parallel_executable.split(' ')
        args.extend(['-n', f'{ntasks}',
                     'ESMF_RegridWeightGen',
                     '-s', slm_scripfile,
                     '-d', mali_scripfile,
                     '-w', 'mapfile_slm_to_mali.nc',
                     '-m', 'bilinear',
                     '-i', '-64bit_offset', '--netcdf4',
                     '--dst_regional'])

        check_call(args, logger)

        logger.info('mapping file creation complete')
