from compass.testgroup import TestGroup
from compass.landice.tests.eismint2.standard_experiments import \
    StandardExperiments
from compass.landice.tests.eismint2.decomposition_test import DecompositionTest
from compass.landice.tests.eismint2.restart_test import RestartTest


class Eismint2(TestGroup):
    """
    A test group for eismint2 test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='eismint2')

        StandardExperiments(test_group=self)

        for thermal_solver in ['temperature', 'enthalpy']:
            DecompositionTest(test_group=self, thermal_solver=thermal_solver)
            RestartTest(test_group=self, thermal_solver=thermal_solver)
