from compass.testcase import TestCase
from compass.mesh.spherical import IcosahedralMeshStep
from compass.ocean.tests.tides.configure import configure_tides
from compass.ocean.mesh.cull import CullMeshStep


class Mesh(TestCase):
    """
    A test case for creating a global MPAS-Ocean mesh
    """
    def __init__(self, test_group, mesh_name):
        """
        Create test case for creating a global MPAS-Ocean mesh

        Parameters
        ----------
        test_group : compass.ocean.tests.tides.Tides
            The test group that this test case belongs to

        mesh_name : str
            The name of the mesh
        """
        self.mesh_name = mesh_name
        name = 'mesh'
        subdir = '{}/{}'.format(mesh_name, name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        name = 'base_mesh'
        if mesh_name == 'Icos7':
            base_mesh_step = IcosahedralMeshStep(
                self, name=name, subdivisions=7)
            mesh_lower = 'icos7'
        else:
            raise ValueError(f'Unexpected mesh name {mesh_name}')

        self.package = f'compass.ocean.tests.tides.mesh.{mesh_lower}'
        self.mesh_config_filename = f'{mesh_lower}.cfg'

        self.add_step(base_mesh_step)

        self.add_step(CullMeshStep(
            test_case=self, base_mesh_step=base_mesh_step,
            with_ice_shelf_cavities=True))

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        configure_tides(test_case=self, mesh=self)
