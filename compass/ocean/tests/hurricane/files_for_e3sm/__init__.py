import os
from datetime import datetime

from compass.ocean.tests.hurricane.configure import configure_hurricane
from compass.ocean.tests.hurricane.files_for_e3sm.domain_files import (
    DomainFiles,
)
from compass.ocean.tests.hurricane.files_for_e3sm.forcing_maps import (
    ForcingMaps,
)
from compass.testcase import TestCase


class FilesForE3SM(TestCase):
    """
    A test case for assembling files needed for MPAS-Ocean forcing in E3SM

    Attributes
    ----------
    mesh : compass.ocean.tests.hurricane.mesh.Mesh
        The test case that produces the mesh for this run
    """
    def __init__(self, test_group, mesh=None):
        """
        Create test case for creating a global MPAS-Ocean mesh

        Parameters
        ----------
        test_group : compass.ocean.tests.hurricane.Hurricane
            The global ocean test group that this test case belongs to

        mesh : compass.ocean.tests.hurricane.mesh.Mesh, optional
            The test case that produces the mesh for this run
        """

        name = 'files_for_e3sm'
        subdir = os.path.join(mesh.mesh_name, name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)
        self.mesh = mesh
        self.creation_date = datetime.now().strftime('%Y%m%d')
        self.add_step(ForcingMaps(test_case=self))
        self.add_step(DomainFiles(test_case=self))

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        configure_hurricane(test_case=self, mesh=self.mesh)
