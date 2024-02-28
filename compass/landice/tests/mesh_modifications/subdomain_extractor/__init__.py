import sys
from importlib import resources

import numpy as np

from compass.landice.tests.mesh_modifications.subdomain_extractor.extract_region import (  # noqa
    ExtractRegion,
)
from compass.testcase import TestCase


class SubdomainExtractor(TestCase):
    """
    A class for a test case that extracts a subdomain from a larger domain
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.mesh_modifications.MeshModifications
            The test group that this test case belongs to

        """
        name = 'subdomain_extractor'
        super().__init__(test_group=test_group, name=name)

        self.add_step(ExtractRegion(test_case=self))

    # no configure() method is needed

    # no run() method is needed

    # no validate() method is needed
