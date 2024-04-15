#from compass.ocean.tests.global_ocean.metadata import (
#    get_author_and_email_from_git,
#)
from compass.testcase import TestCase

#from compass.validate import compare_variables


class WaveMesh(TestCase):
    """
    A test case for creating a global MPAS-Ocean mesh

    Attributes
    ----------
    """
    def __init__(self, test_group, ocean_mesh):
        """
        Create test case for creating a global MPAS-Ocean mesh

        Parameters
        ----------
        test_group : compass.ocean.tests.global_ocean.GlobalOcean
            The global ocean test group that this test case belongs to

        """
        name = 'wave_mesh'
        subdir = f'{mesh_name}/{name}'
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.package = f'compass.ocean.tests.global_ocean.mesh.{mesh_lower}'
        self.mesh_config_filename = f'{mesh_lower}.cfg'

        self.add_step(base_mesh_step)

#    def configure(self, config=None):
#        """
#        Modify the configuration options for this test case
#
#        config : compass.config.CompassConfigParser, optional
#            Configuration options to update if not those for this test case
#        """
#        if config is None:
#            config = self.config
#        config.add_from_package('compass.mesh', 'mesh.cfg', exception=True)
#
#        get_author_and_email_from_git(config)

#    def validate(self):
#        """
#        Test cases can override this method to perform validation of variables
#        and timers
#        """
#        variables = ['xCell', 'yCell', 'zCell']
#        compare_variables(test_case=self, variables=variables,
#                          filename1='cull_mesh/culled_mesh.nc')
