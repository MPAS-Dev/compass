from compass.testgroup import TestGroup
from compass.landice.tests.enthalpy_benchmark.A import A
from compass.landice.tests.enthalpy_benchmark.B import B


class EnthalpyBenchmark(TestGroup):
    """
    A test group for enthalpy benchmark test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='enthalpy_benchmark')

        self.add_test_case(A(test_group=self))
        self.add_test_case(B(test_group=self))
