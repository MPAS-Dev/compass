from compass.testgroup import TestGroup

from compass.ocean.tests.sphere_transport.rotation2D import Rotation2D
from compass.ocean.tests.sphere_transport.nondivergent2D import Nondivergent2D


class SphereTransport(TestGroup):
    """
    A test group for testing algorithms for passive tracer advection on the sphere
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.ocean.Ocean
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='sphere_transport')

        self.add_test_case(Rotation2D(test_group=self))
        self.add_test_case(Nondivergent2D(test_group=self))
