from compass.testcase import TestCase
from compass.ocean.tests.hurricane.mesh.dequ120at30cr10rr2 \
    import DEQU120at30cr10rr2Mesh


class Mesh(TestCase):
    """
    A test case for creating a global MPAS-Ocean mesh

    Attributes
    ----------
    mesh_step : compass.ocean.tests.global_ocean.mesh.mesh.MeshStep
        The step for creating the mesh
    """
    def __init__(self, test_group, mesh_name):
        """
        Create test case for creating a global MPAS-Ocean mesh

        Parameters
        ----------
        test_group : compass.ocean.tests.hurricane.Hurricane
            The test group that this test case belongs to

        mesh_name : str
            The name of the mesh
        """
        self.mesh_name = mesh_name
        name = 'mesh'
        subdir = '{}/{}'.format(mesh_name, name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)
        if mesh_name == 'DEQU120at30cr10rr2':
            self.mesh_step = DEQU120at30cr10rr2Mesh(
                                 self, mesh_name,
                                 preserve_floodplain=False)
        elif mesh_name == 'DEQU120at30cr10rr2WD':
            self.mesh_step = DEQU120at30cr10rr2Mesh(
                                 self, mesh_name,
                                 preserve_floodplain=True)
        else:
            raise ValueError(f'Unexpected mesh name {mesh_name}')

        self.add_step(self.mesh_step)

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        self.config.add_from_package(self.mesh_step.package,
                                     self.mesh_step.mesh_config_filename,
                                     exception=True)

    def run(self):
        """
        Run each step of the testcase
        """
        step = self.mesh_step
        config = self.config

        # get the these properties from the config options
        step.cpus_per_task = config.getint(
            'global_ocean', 'mesh_cpus_per_task')
        step.min_cpus_per_task = config.getint(
            'global_ocean', 'mesh_min_cpus_per_task')

        # run the step
        super().run()
