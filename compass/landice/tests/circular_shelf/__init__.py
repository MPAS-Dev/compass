from compass.testgroup import TestGroup
from compass.landice.tests.circular_shelf.decomposition_test \
    import DecompositionTest


class CircularShelf(TestGroup):
    """
    A test group for circular shelf test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='circular_shelf')

        for mesh_type in ['1250m', ]:
            self.add_test_case(DecompositionTest(test_group=self,
                               mesh_type=mesh_type))
