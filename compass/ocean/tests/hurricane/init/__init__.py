import os

from compass.testcase import TestCase
from compass.ocean.tests.hurricane.init.initial_state import InitialState
from compass.ocean.tests.hurricane.init.create_pointstats_file \
    import CreatePointstatsFile
from compass.ocean.tests.hurricane.init.interpolate_atm_forcing \
    import InterpolateAtmForcing
from compass.ocean.tests.hurricane.configure import configure_hurricane


class Init(TestCase):
    """
    A test case for creating initial conditions on a global MPAS-Ocean mesh

    Attributes
    ----------
    mesh : compass.ocean.tests.hurricane.mesh.Mesh
        The test case that creates the mesh used by this test case
    """
    def __init__(self, test_group, mesh, storm):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.hurricane.Hurricane
            The hurricane test group that this test case belongs to

        mesh : compass.ocean.tests.hurricane.mesh.Mesh
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
        configure_hurricane(test_case=self, mesh=self.mesh)
