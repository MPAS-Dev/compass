import sys
from importlib import resources

import numpy as np

from compass.landice.tests.mesh_modifications.subdomain_extractor.extract_region \  # noqa
    import ExtractRegion
from compass.testcase import TestCase


class SubdomainExtractor(TestCase):
    """
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ensemble_generator.EnsembleGenerator
            The test group that this test case belongs to

        """
        name = 'subdomain_extractor'
        super().__init__(test_group=test_group, name=name)

        self.add_step(ExtractRegion(test_case=self))

    def configure(self):
        """
        """

    # no run() method is needed

    # no validate() method is needed
