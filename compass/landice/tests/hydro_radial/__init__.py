from compass.testgroup import TestGroup
from compass.landice.tests.hydro_radial.decomposition_test import \
    DecompositionTest
from compass.landice.tests.hydro_radial.restart_test import RestartTest
from compass.landice.tests.hydro_radial.spinup_test import SpinupTest
from compass.landice.tests.hydro_radial.steady_state_drift_test import \
    SteadyStateDriftTest


class HydroRadial(TestGroup):
    """
    A test group for radially symmetric hydrology test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='hydro_radial')

        DecompositionTest(test_group=self)
        RestartTest(test_group=self)
        SpinupTest(test_group=self)
        SteadyStateDriftTest(test_group=self)
