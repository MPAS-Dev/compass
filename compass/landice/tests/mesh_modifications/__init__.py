from compass.landice.tests.mesh_modifications.subdomain_extractor import (
    SubdomainExtractor,
)
from compass.testgroup import TestGroup


class MeshModifications(TestGroup):
    """
    A test group for automating modifications to existing meshes
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core,
                         name='mesh_modifications')

        self.add_test_case(SubdomainExtractor(test_group=self))
