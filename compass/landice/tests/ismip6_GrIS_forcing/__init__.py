import yaml

from compass.landice.tests.ismip6_GrIS_forcing.forcing_gen import ForcingGen
from compass.testgroup import TestGroup


class Ismip6GrISForcing(TestGroup):
    """
    A test group for processing the forcing for ISMIP6 GrIS projections
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='ismip6_GrIS_forcing')

        self.add_test_case(ForcingGen(test_group=self))

        with open("experiments.yml", "r") as f:
            experiments = yaml.safe_load(f)

        # should I filter through the experiments dictionary to make sure
        # everything is valid....

        self.experiments = experiments

    def __read_and_validate_experiment_file(self):
        pass
