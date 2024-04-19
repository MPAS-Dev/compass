from mpas_tools.logging import check_call

from compass import Step


class WavesScripFile(Step):
    """
    A step for creating the scrip file for the wave mesh
    """
    def __init__(self, test_case, wave_culled_mesh,
                 name='scrip_file', subdir=None):

        super().__init__(test_case=test_case, name=name, subdir=subdir)

        wave_culled_mesh_path = wave_culled_mesh.path
        self.add_input_file(
            filename='wave_mesh_culled.msh',
            work_dir_target=f'{wave_culled_mesh_path}/wave_mesh_culled.msh')

    def setup(self):

        super().setup()

        f = open(f'{self.work_dir}/scrip.nml', 'w')
        f.write('&inputs\n')
        f.write("    waves_mesh_file = 'wave_mesh_culled.msh'\n")
        f.write("/\n")
        f.write('&outputs\n')
        f.write("    waves_scrip_file = 'wave_mesh_scrip.nc'\n")
        f.write("/\n")
        f.close()

    def run(self):
        """
        Create scrip files for wave mesh
        """
        check_call('ocean_scrip_wave_mesh', logger=self.logger)
