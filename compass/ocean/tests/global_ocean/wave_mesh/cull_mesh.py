from importlib.resources import read_text

from jinja2 import Template
from mpas_tools.logging import check_call

from compass import Step


class WavesCullMesh(Step):
    """
    A step for creating wave mesh based on an ocean mesh
    """
    def __init__(self, test_case, ocean_mesh, wave_base_mesh,
                 name='cull_mesh', subdir=None):

        super().__init__(test_case=test_case, name=name, subdir=subdir)

        self.ocean_mesh = ocean_mesh

        wave_base_mesh_path = wave_base_mesh.path
        self.add_input_file(
            filename='wave_base_mesh.nc',
            work_dir_target=f'{wave_base_mesh_path}/base_mesh.nc')

    def setup(self):

        super().setup()

        if self.config.has_option('wave_mesh', 'ocean_culled_mesh'):
            culled_mesh_path = self.config.get('wave_mesh',
                                               'ocean_culled_mesh')
        else:
            mesh_path = self.ocean_mesh.steps['initial_state'].path
            culled_mesh_path = f'{mesh_path}/initial_state.nc'

        self.add_input_file(
            filename='ocean_culled_mesh.nc',
            work_dir_target=culled_mesh_path)

        template = Template(read_text(
            'compass.ocean.tests.global_ocean.wave_mesh',
            'cull_mesh.template'))

        text = template.render()
        text = f'{text}\n'

        with open(f'{self.work_dir}/cull_waves_mesh.nml', 'w') as file:
            file.write(text)

    def run(self):

        check_call('ocean_cull_wave_mesh', logger=self.logger)
