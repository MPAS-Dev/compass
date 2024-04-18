import os
import pprint
import subprocess

import yaml

from compass import Step


class WavesRemapFiles(Step):
    """
    A step for creating remapping files for wave mesh
    """
    def __init__(self, test_case, wave_scrip, ocean_scrip,
                 name='remap_files', subdir=None):

        super().__init__(test_case=test_case, name=name, subdir=subdir)

        wave_scrip_file_path = wave_scrip.path
        self.add_input_file(
            filename='wave_scrip.nc',
            work_dir_target=f'{wave_scrip_file_path}/wave_scrip.nc')

        ocean_scrip_file_path = ocean_scrip.path
        self.add_input_file(
            filename='ocean_scrip.nc',
            work_dir_target=f'{ocean_scrip_file_path}/ocean_scrip.nc')

    def setup(self):

        super().setup()
        # TO DO: mesh names should be flexible not hard-coded.
        f = open(f'{self.work_dir}/make_remapping_files.config', 'w')
        f.write("grid1 : 'scrip.nc'\n")
        f.write("grid1_shortname : 'wQU225IcoswISC30E3r5'\n")
        f.write("reg1 : True\n")
        f.write("\n")
        f.write("grid2 : 'ocean.IcoswISC30E3r5.mask.scrip.20231120.nc'\n")
        f.write("grid2_shortname : 'IcoswISC30E3r5'\n")
        f.write("reg2 : True\n")
        f.write("\n")
        f.write("datestamp : 20240411\n")
        f.write("\n")
        f.write("map_types : ['conserve','bilinear','neareststod']\n")
        f.close()

    def make_remapping_files(grid1, grid2,
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

        map_name = 'map_' + grid1_shortname + '_TO_' + grid2_shortname +\
                   '_' + map_abbrev + '.' + str(datestamp) + '.nc'
        subprocess.call('srun -n ' + str(nprocs) +
                        ' ESMF_RegridWeightGen --source ' + grid1 +
                        ' --destination ' + grid2 +
                        ' --method ' + map_type +
                        ' --weight ' + map_name +
                        flags, shell=True)

        flags = ' --ignore_unmapped --ignore_degenerate'
        if reg1:
            flags += ' --dst_regional'
        if reg2:
            flags += ' --src_regional'

        map_name = 'map_' + grid2_shortname + '_TO_' + grid1_shortname +\
                   '_' + map_abbrev + '.' + str(datestamp) + '.nc'
        subprocess.call('srun -n ' + str(nprocs) +
                        ' ESMF_RegridWeightGen --source ' + grid2 +
                        ' --destination ' + grid1 +
                        ' --method ' + map_type +
                        ' --weight ' + map_name +
                        flags, shell=True)

    def run(self):
        pwd = os.getcwd()
        f = open(pwd + '/make_remapping_files.config')
        cfg = yaml.load(f, Loader=yaml.Loader)
        pprint.pprint(cfg)
        for map_type in cfg['map_types']:
            self.make_remapping_files(cfg['grid1'],
                                      cfg['grid2'],
                                      cfg['grid1_shortname'],
                                      cfg['grid2_shortname'],
                                      cfg['datestamp'],
                                      cfg['reg1'], cfg['reg2'],
                                      map_type)
