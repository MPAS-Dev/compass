from compass.testgroup import TestGroup
from compass.landice.tests.greenland.smoke_test import SmokeTest
from compass.landice.tests.greenland.decomposition_test import DecompositionTest
from compass.landice.tests.greenland.restart_test import RestartTest


class Greenland(TestGroup):
    """
    A test group for Greenland test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='greenland')

        SmokeTest(test_group=self)
        DecompositionTest(test_group=self)
        RestartTest(test_group=self)
