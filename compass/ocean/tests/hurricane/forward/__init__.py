import os

from compass.ocean.tests.hurricane.analysis import Analysis
from compass.ocean.tests.hurricane.configure import configure_hurricane
from compass.ocean.tests.hurricane.forward.forward import ForwardStep
from compass.testcase import TestCase


class Forward(TestCase):
    """
    A test case for performing a forward run for a hurricane case

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.Mesh
        The test case that produces the mesh for this run

    init : compass.ocean.tests.hurricane.init.Init
        The test case that produces the initial condition for this run
    """
    def __init__(self, test_group, mesh, storm, init, use_lts):
        """
        Create test case

        Parameters
        ----------
        test_group : compass.ocean.tests.hurricane.Hurricane
            The test group that this test case belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that produces the mesh for this run

        storm : str
            The name of the storm to be run

        init : compass.ocean.tests.hurricane.init.Init
            The test case that produces the initial condition for this run

        use_lts : bool
            Whether local time-stepping is to be used
        """
        mesh_name = mesh.mesh_name

        if use_lts == 'LTS':
            name = f'{storm}_lts'
        elif use_lts == 'FB_LTS':
            name = f'{storm}_fblts'
        else:
            name = storm
        subdir = os.path.join(mesh_name, name)
        super().__init__(test_group=test_group,
                         subdir=subdir,
                         name=name)
        self.mesh = mesh

        step = ForwardStep(test_case=self,
                           mesh=mesh,
                           init=init,
                           use_lts=use_lts)

        step.add_output_file(filename='output.nc')
        step.add_output_file(filename='pointwiseStats.nc')
        self.add_step(step)

        step = Analysis(test_case=self, storm=storm)
        self.add_step(step)

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        configure_hurricane(test_case=self, mesh=self.mesh)
