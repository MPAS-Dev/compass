import os

from compass.ocean.tests.hurricane_lts.configure import configure_hurricane_lts
from compass.ocean.tests.hurricane_lts.init.create_pointstats_file import (
    CreatePointstatsFile,
)
from compass.ocean.tests.hurricane_lts.init.initial_state import InitialState
from compass.ocean.tests.hurricane_lts.init.interpolate_atm_forcing import (
    InterpolateAtmForcing,
)
from compass.testcase import TestCase


class Init(TestCase):
    """
    A test case for creating initial conditions on a global MPAS-Ocean mesh

    Attributes
    ----------
    mesh : compass.ocean.tests.hurricane_lts.mesh.Mesh
        The test case that creates the mesh used by this test case
    """
    def __init__(self, test_group, mesh, storm):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.hurricane_lts.Hurricane_LTS
            The hurricane with local time-stepping test group
            that this test case belongs to

        mesh : compass.ocean.tests.hurricane_lts.mesh.Mesh
            The test case that creates the mesh used by this test case

        storm : str
            The name of the storm to be run
        """
        name = 'init'
        mesh_name = mesh.mesh_name
        subdir = os.path.join(mesh_name, name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.mesh = mesh

        self.add_step(InitialState(test_case=self, mesh=mesh))
        self.add_step(InterpolateAtmForcing(test_case=self, mesh=mesh,
                                            storm=storm))
        self.add_step(CreatePointstatsFile(test_case=self, mesh=mesh,
                                           storm=storm))

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        configure_hurricane_lts(test_case=self, mesh=self.mesh)
