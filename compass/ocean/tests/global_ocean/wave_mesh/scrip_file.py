from importlib.resources import read_text

from jinja2 import Template
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

        template = Template(read_text(
            'compass.ocean.tests.global_ocean.wave_mesh',
            'scrip_file.template'))

        text = template.render()
        text = f'{text}\n'

        with open(f'{self.work_dir}/scrip.nml', 'w') as file:
            file.write(text)

    def run(self):
        """
        Create scrip files for wave mesh
        """
        check_call('ocean_scrip_wave_mesh', logger=self.logger)
