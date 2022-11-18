from compass.testcase import TestCase
from compass.mesh.spherical import IcosahedralMeshStep, \
    QuasiUniformSphericalMeshStep
from compass.ocean.mesh.cull import CullMeshStep
from compass.ocean.tests.global_ocean.mesh.arrm10to60 import ARRM10to60BaseMesh
from compass.ocean.tests.global_ocean.mesh.ec30to60 import EC30to60BaseMesh
from compass.ocean.tests.global_ocean.mesh.so12to60 import SO12to60BaseMesh
from compass.ocean.tests.global_ocean.mesh.wc14 import WC14BaseMesh
from compass.ocean.tests.global_ocean.configure import configure_global_ocean
from compass.validate import compare_variables


class Mesh(TestCase):
    """
    A test case for creating a global MPAS-Ocean mesh

    Attributes
    ----------
    package : str
        The python package for the mesh

    mesh_config_filename : str
        The name of the mesh config file

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
        subdir = f'{mesh_name}/{name}'
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        with_ice_shelf_cavities = 'wISC' in mesh_name
        mesh_lower = mesh_name.lower()
        if with_ice_shelf_cavities:
            mesh_lower = mesh_lower.replace('wisc', '')
        if 'icos' in mesh_lower:
            mesh_lower = mesh_lower.replace('icos', 'qu')

        self.package = f'compass.ocean.tests.global_ocean.mesh.{mesh_lower}'
        self.mesh_config_filename = f'{mesh_lower}.cfg'

        self.mesh_name = mesh_name
        self.with_ice_shelf_cavities = with_ice_shelf_cavities

        name = 'base_mesh'
        subdir = None
        if mesh_name in ['Icos240', 'IcoswISC240']:
            base_mesh_step = IcosahedralMeshStep(
                self, name=name, subdir=subdir, cell_width=240)
        elif mesh_name in ['QU240', 'QUwISC240']:
            base_mesh_step = QuasiUniformSphericalMeshStep(
                self, name=name, subdir=subdir, cell_width=240)
        elif mesh_name in ['EC30to60', 'ECwISC30to60']:
            base_mesh_step = EC30to60BaseMesh(self, name=name, subdir=subdir)
        elif mesh_name in ['ARRM10to60']:
            base_mesh_step = ARRM10to60BaseMesh(self, name=name, subdir=subdir)
        elif mesh_name in ['SO12to60', 'SOwISC12to60']:
            base_mesh_step = SO12to60BaseMesh(self, name=name, subdir=subdir)
        elif mesh_name in ['WC14']:
            base_mesh_step = WC14BaseMesh(self, name=name, subdir=subdir)
        else:
            raise ValueError(f'Unknown mesh name {mesh_name}')

        self.add_step(base_mesh_step)

        self.add_step(CullMeshStep(
            test_case=self, base_mesh_step=base_mesh_step,
            with_ice_shelf_cavities=self.with_ice_shelf_cavities))

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        configure_global_ocean(test_case=self, mesh=self)
        config = self.config
        config.set('spherical_mesh', 'add_mesh_density', 'True')
        config.set('spherical_mesh', 'plot_cell_width', 'True')

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['xCell', 'yCell', 'zCell']
        compare_variables(test_case=self, variables=variables,
                          filename1='cull_mesh/culled_mesh.nc')

    def get_cull_mesh_path(self):
        """
        Get the path of the cull mesh step (for input files)
        Returns
        -------
        cull_mesh_path : str
            The path to the work directory of the cull mesh step.
        """
        return self.steps['cull_mesh'].path
