from mpas_tools.logging import check_call

from compass import Step


class WavesRotateMesh(Step):
    """
    A step for creating wave mesh based on an ocean mesh
    """
    def __init__(self, test_case, wave_culled_mesh,
                 name='rotate_mesh', subdir=None):

        super().__init__(test_case=test_case, name=name, subdir=subdir)

        wave_culled_mesh_path = wave_culled_mesh.path
        self.add_input_file(
            filename='wave_mesh_culled.msh',
            work_dir_target=f'{wave_culled_mesh_path}/wave_mesh_culled.msh')

    def setup(self):

        super().setup()

        f = open(f'{self.work_dir}/rotate.nml', 'w')
        f.write('&inputs\n')
        f.write("    LON_POLE = -42.8906d0\n")
        f.write("    LAT_POLE = 72.3200d0\n")
        f.write("    wind_file = 'null'\n")
        f.write("    mesh_file = 'wave_mesh_culled.msh'\n")
        f.write("    mesh_file_out = 'wave_culled_mesh_RTD.msh'\n")
        f.write("/\n")
        f.close()

    def run(self):

        check_call('ocean_rotate_wave_mesh', logger=self.logger)
