from compass.testgroup import TestGroup

from compass.ocean.tests.sphere_transport.rotation_2d import Rotation2D
from compass.ocean.tests.sphere_transport.nondivergent_2d import Nondivergent2D
from compass.ocean.tests.sphere_transport.divergent_2d import Divergent2D
from compass.ocean.tests.sphere_transport.correlated_tracers_2d import \
    CorrelatedTracers2D


class SphereTransport(TestGroup):
    """
    A test group for testing algorithms for passive tracer advection
    on the sphere
    """

    def __init__(self, mpas_core):
        """
        mpas_core : compass.ocean.Ocean
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='sphere_transport')

        self.add_test_case(Rotation2D(test_group=self))
        self.add_test_case(Nondivergent2D(test_group=self))
        self.add_test_case(Divergent2D(test_group=self))
        self.add_test_case(CorrelatedTracers2D(test_group=self))
