import os

from compass.testcase import TestCase
from compass.ocean.tests.hurricane.init.initial_state import InitialState
from compass.ocean.tests.hurricane.init.create_pointstats_file \
    import CreatePointstatsFile
from compass.ocean.tests.hurricane.init.interpolate_atm_forcing \
    import InterpolateAtmForcing
from compass.ocean.tests.global_ocean.configure import configure_global_ocean


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

        """
        name = 'init'
        mesh_name = mesh.mesh_name
        subdir = os.path.join(mesh_name, name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.mesh = mesh
        self.name = name

        self.add_step(InitialState(test_case=self, mesh=mesh))
        self.add_step(InterpolateAtmForcing(test_case=self, mesh=mesh,
                                            storm=storm))
        self.add_step(CreatePointstatsFile(test_case=self, mesh=mesh,
                                           storm=storm))

        self.initial_condition = 'PHC'
        self.with_bgc = False

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        configure_global_ocean(test_case=self, mesh=self.mesh, init=self)

    def run(self):
        """
        Run each step of the testcase
        """

        # run the steps
        super().run()
