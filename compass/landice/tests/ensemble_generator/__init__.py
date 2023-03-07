from compass.landice.tests.ensemble_generator.ensemble import Ensemble
from compass.testgroup import TestGroup


class EnsembleGenerator(TestGroup):
    """
    A test group for generating ensembles of MALI simulations
    for uncertainty quantification or parameter sensitivity tests.
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core,
                         name='ensemble_generator')

        self.add_test_case(Ensemble(test_group=self))
