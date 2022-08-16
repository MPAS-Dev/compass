from compass.testcase import TestCase
from compass.ocean.tests.global_ocean.mesh.qu240 import QU240Mesh
from compass.ocean.tests.global_ocean.mesh.ec30to60 import EC30to60Mesh
from compass.ocean.tests.global_ocean.mesh.so12to60 import SO12to60Mesh
from compass.ocean.tests.global_ocean.mesh.wc14 import WC14Mesh
from compass.ocean.tests.global_ocean.configure import configure_global_ocean
from compass.validate import compare_variables


class Mesh(TestCase):
    """
    A test case for creating a global MPAS-Ocean mesh

    Attributes
    ----------
    mesh_step : compass.ocean.tests.global_ocean.mesh.mesh.MeshStep
        The step for creating the mesh

    with_ice_shelf_cavities : bool
        Whether the mesh includes ice-shelf cavities
    """
    def __init__(self, test_group, mesh_name):
        """
        Create test case for creating a global MPAS-Ocean mesh

        Parameters
        ----------
        test_group : compass.ocean.tests.global_ocean.GlobalOcean
            The global ocean test group that this test case belongs to

        mesh_name : str
            The name of the mesh
        """
        name = 'mesh'
        subdir = '{}/{}'.format(mesh_name, name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)
        if mesh_name in 'QU240':
            self.mesh_step = QU240Mesh(self, mesh_name,
                                       with_ice_shelf_cavities=False)
        elif mesh_name in 'QUwISC240':
            self.mesh_step = QU240Mesh(self, mesh_name,
                                       with_ice_shelf_cavities=True)
        elif mesh_name in 'EC30to60':
            self.mesh_step = EC30to60Mesh(self, mesh_name,
                                          with_ice_shelf_cavities=False)
        elif mesh_name in 'ECwISC30to60':
            self.mesh_step = EC30to60Mesh(self, mesh_name,
                                          with_ice_shelf_cavities=True)
        elif mesh_name in 'SOwISC12to60':
            self.mesh_step = SO12to60Mesh(self, mesh_name,
                                          with_ice_shelf_cavities=True)
        elif mesh_name in 'WC14':
            self.mesh_step = WC14Mesh(self, mesh_name,
                                      with_ice_shelf_cavities=False)
        else:
            raise ValueError('Unknown mesh name {}'.format(mesh_name))

        self.add_step(self.mesh_step)

        self.mesh_name = mesh_name
        self.with_ice_shelf_cavities = self.mesh_step.with_ice_shelf_cavities

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        configure_global_ocean(test_case=self, mesh=self)

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

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['xCell', 'yCell', 'zCell']
        compare_variables(test_case=self, variables=variables,
                          filename1='mesh/culled_mesh.nc')
