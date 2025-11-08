import os

import yaml

from compass.landice.tests.ismip6_GrIS_forcing.forcing_gen import ForcingGen
from compass.testgroup import TestGroup


class Ismip6GrISForcing(TestGroup):
    """
    A test group for processing the forcing for ISMIP6 GrIS projections
    """
    def __init__(self, mpas_core):
        """
        Parameters
        ----------
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='ismip6_GrIS_forcing')

        self.add_test_case(ForcingGen(test_group=self))

        # open, read, and validate the experiment file
        self._read_and_validate_experiment_file()

    def _read_and_validate_experiment_file(self):

        # get the filepath to current module, needed of opening experiment file
        module_path = os.path.dirname(os.path.realpath(__file__))

        with open(f"{module_path}/experiments.yml", "r") as f:
            experiments = yaml.safe_load(f)

        # experiments dictionary is unverified...
        # But the yaml file packages with compass shouldn't really be altered
        self.experiments = experiments
