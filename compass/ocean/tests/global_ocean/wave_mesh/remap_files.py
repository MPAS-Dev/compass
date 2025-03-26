from datetime import date
from distutils.spawn import find_executable

from mpas_tools.logging import check_call

from compass import Step


class WavesRemapFiles(Step):
    """
    A step for creating remapping files for wave mesh
    """
    def __init__(self, test_case, wave_scrip,
                 name='remap_files', subdir=None):

        super().__init__(test_case=test_case, name=name, subdir=subdir)

        wave_scrip_file_path = wave_scrip.path
        self.add_input_file(
            filename='wave_scrip.nc',
            work_dir_target=f'{wave_scrip_file_path}/wave_mesh_scrip.nc')

        ocean_scrip_file_path = wave_scrip.path
        self.add_input_file(
            filename='ocean_scrip.nc',
            work_dir_target=f'{ocean_scrip_file_path}/ocean_mesh_scrip.nc')

    def make_remapping_files(self, grid1, grid2,
                             grid1_shortname, grid2_shortname,
                             datestamp, reg1, reg2, map_type, nprocs=1):

        if map_type == 'conserve':
            map_abbrev = 'aave'
        elif map_type == 'bilinear':
            map_abbrev = 'blin'
        elif map_type == 'patch':
            map_abbrev = 'patc'
        elif map_type == 'neareststod':
            map_abbrev = 'nstod'
        elif map_type == 'nearsetdtos':
            map_abbrev = 'ndtos'
        else:
            print('map type not recognized')
            raise SystemExit(0)

        exe = find_executable('ESMF_RegridWeightGen')
        parallel_executable = self.config.get('parallel',
                                              'parallel_executable')
        ntasks = self.ntasks
        parallel_args = parallel_executable.split(' ')
        if 'srun' in parallel_args:
            parallel_args.extend(['-n', f'{ntasks}'])
        else:  # presume mpirun syntax
            parallel_args.extend(['-np', f'{ntasks}'])

        flags = ['--ignore_unmapped', '--ignore_degenerate']

        # wave to ocean remap
        map_name = f'map_{grid1_shortname}_TO_{grid2_shortname}'\
                   f'_{map_abbrev}.{datestamp}.nc'

        args = [exe,
                '--source', grid1,
                '--destination', grid2,
                '--method', map_type,
                '--weight', map_name]
        args.extend(flags)
        if reg1:
            args.append('--src_regional')
        if reg2:
            args.append('--dst_regional')

        cmd = parallel_args + args
        check_call(cmd, logger=self.logger)

        # ocean to wave remap
        map_name = f'map_{grid2_shortname}_TO_{grid1_shortname}'\
                   f'_{map_abbrev}.{datestamp}.nc'

        args = [exe,
                '--source', grid2,
                '--destination', grid1,
                '--method', map_type,
                '--weight', map_name]
        args.extend(flags)
        if reg1:
            args.append('--dst_regional')
        if reg2:
            args.append('--src_regional')

        cmd = parallel_args + args
        check_call(cmd, logger=self.logger)

    def run(self):
        today = date.today()
        creation_date = today.strftime("%Y%m%d")
        for map_type in ['conserve', 'bilinear', 'neareststod']:
            self.make_remapping_files('wave_scrip.nc',
                                      'ocean_scrip.nc',
                                      'wave',   # This should be more specific
                                      'ocean',  # This should be more spedific
                                      creation_date,
                                      True, True,
                                      map_type)
