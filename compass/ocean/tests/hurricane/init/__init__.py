import os

from compass.ocean.tests.hurricane.configure import configure_hurricane
from compass.ocean.tests.hurricane.init.create_pointstats_file import (
    CreatePointstatsFile,
)
from compass.ocean.tests.hurricane.init.initial_state import InitialState
from compass.ocean.tests.hurricane.init.interpolate_atm_forcing import (
    InterpolateAtmForcing,
)
from compass.ocean.tests.hurricane.lts.init.topographic_wave_drag import (
    ComputeTopographicWaveDrag,
)
from compass.testcase import TestCase


class Init(TestCase):
    """
    A test case for creating initial conditions on a global MPAS-Ocean mesh

    Attributes
    ----------
    mesh : compass.ocean.tests.hurricane.mesh.Mesh
        The test case that creates the mesh used by this test case
    """
    def __init__(self, test_group, mesh, storm, use_lts):
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

        use_lts : bool
            Whether local time-stepping is used
        """

        self.mesh = mesh
        self.use_lts = use_lts

        if use_lts == 'LTS':
            name = 'init_lts'
        elif use_lts == 'FB_LTS':
            name = 'init_fblts'
        else:
            name = 'init'
        mesh_name = mesh.mesh_name
        subdir = os.path.join(mesh_name, name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.add_step(InitialState(test_case=self, mesh=mesh, use_lts=use_lts))
        self.add_step(InterpolateAtmForcing(test_case=self, mesh=mesh,
                                            storm=storm))

        if use_lts:
            topo = ComputeTopographicWaveDrag(test_case=self, mesh=mesh)
            self.add_step(topo)

        self.add_step(CreatePointstatsFile(test_case=self, mesh=mesh,
                                           storm=storm))

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        configure_hurricane(test_case=self, mesh=self.mesh)
