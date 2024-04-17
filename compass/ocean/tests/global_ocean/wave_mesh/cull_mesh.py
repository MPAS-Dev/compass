from mpas_tools.logging import check_call

from compass import Step


class WavesCullMesh(Step):
    """
    A step for creating wave mesh based on an ocean mesh
    """
    def __init__(self, test_case, ocean_mesh, wave_base_mesh,
                 name='cull_mesh', subdir=None):

        super().__init__(test_case=test_case, name=name, subdir=subdir)

        culled_mesh_path = ocean_mesh.steps['cull_mesh'].path
        self.add_input_file(
            filename='ocean_culled_mesh.nc',
            work_dir_target=f'{culled_mesh_path}/culled_mesh.nc')

        wave_base_mesh_path = wave_base_mesh.path
        self.add_input_file(
            filename='wave_base_mesh.nc',
            work_dir_target=f'{wave_base_mesh_path}/wave_base_mesh.nc')

    def setup(self):

        super().setup()

        f = open(f'{self.work_dir}/cull_waves_mesh.nml', 'w')
        f.write('&inputs\n')
        f.write("    waves_mesh_file = 'wave_base_mesh.nc'\n")
        f.write("    ocean_mesh_file = 'ocean_culled_mesh.nc'\n")
        f.write("/\n")
        f.write("&output\n")
        f.write("    waves_mesh_culled_vtk = 'wave_mesh_culled.vtk'\n")
        f.write("    waves_mesh_culled_gmsh = 'wave_mesh_culled.msh'\n")
        f.write("/\n")
        f.close()

    def run(self):

        check_call('ocean_cull_wave_mesh', logger=self.logger)
