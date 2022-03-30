from compass.testgroup import TestGroup

from compass.ocean.tests.spherical_harmonic_transform.qu_convergence \
    import QuConvergence


class SphericalHarmonicTransform(TestGroup):
    """
    A test group for testing spherical harmonic transforms
    used for self attraction and loading calculations in MPAS-Ocean
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.ocean.Ocean
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core,
                         name='spherical_harmonic_transform')

        self.add_test_case(QuConvergence(test_group=self))
