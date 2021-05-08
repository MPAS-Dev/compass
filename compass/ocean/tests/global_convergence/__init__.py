from compass.testgroup import TestGroup

from compass.ocean.tests.global_convergence.cosine_bell import CosineBell


class GlobalConvergence(TestGroup):
    """
    A test group for setting up global initial conditions and performing
    regression testing and dynamic adjustment for MPAS-Ocean
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.ocean.Ocean
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='global_convergence')

        self.add_test_case(CosineBell(test_group=self))
