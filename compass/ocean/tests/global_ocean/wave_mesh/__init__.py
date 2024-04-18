# from compass.ocean.tests.global_ocean.metadata import (
#     get_author_and_email_from_git,
# )
from compass.ocean.tests.global_ocean.wave_mesh.base_mesh import WavesBaseMesh
from compass.ocean.tests.global_ocean.wave_mesh.cull_mesh import WavesCullMesh
from compass.ocean.tests.global_ocean.wave_mesh.rotate_mesh import (
    WavesRotateMesh,
)
from compass.ocean.tests.global_ocean.wave_mesh.uost_files import (
    WavesUostFiles,
)
from compass.testcase import TestCase

# from compass.validate import compare_variables


class WaveMesh(TestCase):
    """
    A test case for creating a global MPAS-Ocean mesh

    Attributes
    ----------
    """
    def __init__(self, test_group, ocean_mesh, ocean_init):
        """
        Create test case for creating a global MPAS-Ocean mesh

        Parameters
        ----------
        test_group : compass.ocean.tests.global_ocean.GlobalOcean
            The global ocean test group that this test case belongs to

        """
        name = 'wave_mesh'
        subdir = f'{ocean_mesh.mesh_name}/{name}'
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.package = 'compass.ocean.tests.global_ocean.wave_mesh'
        self.mesh_config_filename = 'wave_mesh.cfg'

        base_mesh_step = WavesBaseMesh(test_case=self,
                                       ocean_mesh=ocean_init)
        self.add_step(base_mesh_step)

        cull_mesh_step = WavesCullMesh(test_case=self,
                                       ocean_mesh=ocean_init,
                                       wave_base_mesh=base_mesh_step)
        self.add_step(cull_mesh_step)

        rotate_mesh_step = WavesRotateMesh(test_case=self,
                                           wave_culled_mesh=cull_mesh_step)
        self.add_step(rotate_mesh_step)

        uost_file_step = WavesUostFiles(test_case=self,
                                        wave_culled_mesh=cull_mesh_step)
        self.add_step(uost_file_step)

    def configure(self, config=None):
        """
        Modify the configuration options for this test case

        config : compass.config.CompassConfigParser, optional
            Configuration options to update if not those for this test case
        """
        if config is None:
            config = self.config
        config.add_from_package('compass.mesh', 'mesh.cfg', exception=True)

        # get_author_and_email_from_git(config)

#    def validate(self):
#        """
#        Test cases can override this method to perform validation of variables
#        and timers
#        """
#        variables = ['xCell', 'yCell', 'zCell']
#        compare_variables(test_case=self, variables=variables,
#                          filename1='cull_mesh/culled_mesh.nc')
