from datetime import date

from mpas_tools.logging import check_call

from compass import Step


class WavesRemapFiles(Step):
    """
    A step for creating remapping files for wave mesh
    """
    def __init__(self, test_case, wave_scrip, ocean_e3sm,
                 name='remap_files', subdir=None):

        super().__init__(test_case=test_case, name=name, subdir=subdir)

        wave_scrip_file_path = wave_scrip.path
        self.add_input_file(
            filename='wave_scrip.nc',
            work_dir_target=f'{wave_scrip_file_path}/wave_mesh_scrip.nc')

        ocean_scrip_file_path = ocean_e3sm.steps['scrip'].path
        self.add_input_file(
            filename='ocean_scrip.nc',
            work_dir_target=f'{ocean_scrip_file_path}/ocean_scrip.nc')

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

        flags = ' --ignore_unmapped --ignore_degenerate'
        if reg1:
            flags += ' --src_regional'
        if reg2:
            flags += ' --dst_regional'

        map_name = f'map_{grid1_shortname}_TO_{grid2_shortname}'\
                   f'_{map_abbrev}.{datestamp}.nc'
        check_call(f'srun -n {nprocs}'
                   f' ESMF_RegridWeightGen --source {grid1}'
                   f' --destination {grid2}'
                   f' --method {map_type}'
                   f' --weight {map_name} {flags}',
                   logger=self.logger)

        flags = ' --ignore_unmapped --ignore_degenerate'
        if reg1:
            flags += ' --dst_regional'
        if reg2:
            flags += ' --src_regional'

        map_name = f'map_{grid2_shortname}_TO_{grid1_shortname}'\
                   f'_{map_abbrev}.{datestamp}.nc'
        check_call(f'srun -n {nprocs}'
                   f' ESMF_RegridWeightGen --source {grid2}'
                   f' --destination {grid1}'
                   f' --method {map_type}'
                   f' --weight {map_name} {flags}',
                   logger=self.logger)

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
