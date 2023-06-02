from compass.ocean.tests.lock_exchange.hydro import Hydro
from compass.ocean.tests.lock_exchange.nonhydro import Nonhydro
from compass.testgroup import TestGroup


class LockExchange(TestGroup):
    """
    A test group for lock exchange type test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='lock_exchange')

        self.add_test_case(
            Hydro(test_group=self))
        self.add_test_case(
            Nonhydro(test_group=self))
