import os

from compass.testcase import TestCase
from compass.ocean.tests.tides.init.interpolate_wave_drag \
    import InterpolateWaveDrag
from compass.ocean.tests.tides.init.remap_bathymetry \
    import RemapBathymetry
from compass.ocean.tests.tides.init.initial_state import InitialState
from compass.ocean.tests.tides.configure import configure_tides


class Init(TestCase):
    """
    A test case for creating initial conditions on a global MPAS-Ocean mesh

    Attributes
    ----------
    mesh : compass.ocean.tests.tides.mesh.Mesh
        The test case that creates the mesh used by this test case
    """
    def __init__(self, test_group, mesh):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.tides.Tides
            The tides test group that this test case belongs to

        mesh : compass.ocean.tests.tides.mesh.Mesh
            The test case that creates the mesh used by this test case
        """
        name = 'init'
        mesh_name = mesh.mesh_name
        subdir = os.path.join(mesh_name, name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.mesh = mesh

        self.add_step(InterpolateWaveDrag(test_case=self, mesh=mesh))
        self.add_step(RemapBathymetry(test_case=self, mesh=mesh))
        self.add_step(InitialState(test_case=self, mesh=mesh))

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        configure_tides(test_case=self, mesh=self.mesh)
