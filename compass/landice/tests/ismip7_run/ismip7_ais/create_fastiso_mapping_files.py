import shutil
import sys

import netCDF4
import numpy as np
from mpas_tools.logging import check_call
from mpas_tools.scrip.from_mpas import scrip_from_mpas
from mpas_tools.scrip.from_planar import main as scrip_from_planar

from compass.step import Step

# AIS: polar stereographic EPSG:3031, standard parallel 71S, central meridian 0
# Cell centres span -3,040,000 m to 3,040,000 m in both x and y.
AIS_X_MIN = -3040000
AIS_X_MAX = 3040000
AIS_Y_MIN = -3040000
AIS_Y_MAX = 3040000


def create_ismip7_grid_file(icesheet, res_km, output_file):
    """
    Create a minimal ISMIP7 standard grid file containing only x and y
    projected coordinate variables (metres).

    This function is copied from MPAS-Tools to create the regular grid
    for FastIsostasy, which uses the same standard grid as ISMIP7.

    Parameters
    ----------
    icesheet : str
        Ice sheet domain: 'AIS' or 'GrIS'.
    res_km : int
        Grid resolution in kilometres.
    output_file : str
        Path for the output NetCDF file.
    """
    res_m = int(res_km) * 1000
    if icesheet == 'AIS':
        x = np.arange(AIS_X_MIN, AIS_X_MAX + res_m, res_m, dtype=float)
        y = np.arange(AIS_Y_MIN, AIS_Y_MAX + res_m, res_m, dtype=float)
    elif icesheet == 'GrIS':
        # GrIS: polar stereographic EPSG:3413, standard parallel 70N,
        # central meridian 315E
        # Cell centres span x: -720,000 to 960,000 m;
        # y: -3,450,000 to -570,000 m
        GRIS_X_MIN = -720000
        GRIS_X_MAX = 960000
        GRIS_Y_MIN = -3450000
        GRIS_Y_MAX = -570000
        x = np.arange(GRIS_X_MIN, GRIS_X_MAX + res_m, res_m, dtype=float)
        y = np.arange(GRIS_Y_MIN, GRIS_Y_MAX + res_m, res_m, dtype=float)
    else:
        raise ValueError(f"Unknown icesheet '{icesheet}'.")

    ds = netCDF4.Dataset(output_file, 'w')
    ds.createDimension('x', len(x))
    ds.createDimension('y', len(y))
    xv = ds.createVariable('x', 'f8', ('x',))
    yv = ds.createVariable('y', 'f8', ('y',))
    xv[:] = x
    yv[:] = y
    xv.units = 'm'
    xv.standard_name = 'projection_x_coordinate'
    xv.long_name = 'x'
    yv.units = 'm'
    yv.standard_name = 'projection_y_coordinate'
    yv.long_name = 'y'
    ds.close()
    print(f"Created ISMIP7 grid file: {output_file} "
          f"({len(x)} x {len(y)} cells at {res_km} km)")


class CreateFastIsoMappingFiles(Step):
    """
    A step for creating mapping files for FastIsostasy bedrock model
    """

    def __init__(self, test_case, name, subdir):
        """
        Initialize step
        """
        super().__init__(test_case=test_case, name=name, subdir=subdir)

    def setup(self):
        print("    Setting up fastiso_mapping_files subdirectory")

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger
        section = config['ismip7_run_ais']
        fastisostasy = section.getboolean('fastisostasy')
        if fastisostasy:
            self._build_mapping_files(config, logger)

    def _build_mapping_files(self, config, logger):
        """
        Build mapping files between the MALI mesh and the FastIsostasy grid.

        FastIsostasy uses a regular ISMIP7 grid (same as used for output),
        so we create that grid and build bidirectional mapping files.
        """
        section = config['ismip7_run_ais']
        init_cond_path = section.get('init_cond_path')
        fastiso_res_km = section.getint('fastiso_res_km')
        icesheet = section.get('icesheet')
        section = config['parallel']
        ntasks = section.getint('cores_per_node')

        mali_scripfile = 'mali_scripfile.nc'
        fastiso_scripfile = f'fastiso_{fastiso_res_km}km_scripfile.nc'
        mali_meshfile = 'mali_meshfile.nc'
        fastiso_gridfile = f'fastiso_grid_{fastiso_res_km}km.nc'

        # Create FastIsostasy ISMIP7 grid file
        logger.info(f'Creating FastIsostasy grid file with '
                    f'{fastiso_res_km} km resolution')
        create_ismip7_grid_file(icesheet, fastiso_res_km, fastiso_gridfile)

        # Create FastIsostasy scripfile
        logger.info('Creating scripfile for the FastIsostasy grid')

        # Determine projection based on ice sheet
        if icesheet == 'AIS':
            projection = 'ais-bedmap2'
        elif icesheet == 'GrIS':
            projection = 'gis-gimp'
        else:
            raise ValueError(f"Unknown icesheet '{icesheet}'")

        # Use mpas_tools.scrip.from_planar module directly
        # Set up sys.argv to mimic command-line arguments
        old_argv = sys.argv
        sys.argv = ['scrip_from_planar',
                    '--input', fastiso_gridfile,
                    '--scrip', fastiso_scripfile,
                    '--proj', projection,
                    '--rank', '2']

        try:
            scrip_from_planar()
        finally:
            sys.argv = old_argv

        # Create MALI scripfile
        logger.info('Creating scripfile for the MALI mesh')
        shutil.copy(init_cond_path, mali_meshfile)
        scrip_from_mpas(mali_meshfile, mali_scripfile)

        # MALI -> FastIsostasy mapping file (conserve)
        logger.info('Creating MALI -> FastIsostasy grid mapfile '
                    'with conserve method')

        parallel_executable = config.get("parallel", "parallel_executable")
        args = parallel_executable.split(' ')
        args.extend(['-n', f'{ntasks}',
                     'ESMF_RegridWeightGen',
                     '-s', mali_scripfile,
                     '-d', fastiso_scripfile,
                     '-w', 'mapfile_mali_to_fastiso.nc',
                     '-m', 'conserve',
                     '-i', '-64bit_offset', '--netcdf4',
                     '--src_regional', '--dst_regional'])

        check_call(args, logger)

        # FastIsostasy -> MALI mapping file (bilinear)
        logger.info('Creating FastIsostasy -> MALI mesh mapfile '
                    'with bilinear method')
        args = parallel_executable.split(' ')
        args.extend(['-n', f'{ntasks}',
                     'ESMF_RegridWeightGen',
                     '-s', fastiso_scripfile,
                     '-d', mali_scripfile,
                     '-w', 'mapfile_fastiso_to_mali.nc',
                     '-m', 'bilinear',
                     '-i', '-64bit_offset', '--netcdf4',
                     '--src_regional', '--dst_regional'])

        check_call(args, logger)

        logger.info('FastIsostasy mapping file creation complete')
