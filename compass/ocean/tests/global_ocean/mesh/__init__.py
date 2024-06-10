import os

from compass.mesh.spherical import (
    IcosahedralMeshStep,
    QuasiUniformSphericalMeshStep,
)
from compass.ocean.mesh.cull import CullMeshStep
from compass.ocean.mesh.remap_topography import RemapTopography
from compass.ocean.tests.global_ocean.mesh.arrm10to60 import ARRM10to60BaseMesh
from compass.ocean.tests.global_ocean.mesh.ec30to60 import EC30to60BaseMesh
from compass.ocean.tests.global_ocean.mesh.fris01to60 import FRIS01to60BaseMesh
from compass.ocean.tests.global_ocean.mesh.fris02to60 import FRIS02to60BaseMesh
from compass.ocean.tests.global_ocean.mesh.fris04to60 import FRIS04to60BaseMesh
from compass.ocean.tests.global_ocean.mesh.fris08to60 import FRIS08to60BaseMesh
from compass.ocean.tests.global_ocean.mesh.kuroshio import KuroshioBaseMesh
from compass.ocean.tests.global_ocean.mesh.qu import (
    IcosMeshFromConfigStep,
    QUMeshFromConfigStep,
)
from compass.ocean.tests.global_ocean.mesh.remap_mali_topography import (
    RemapMaliTopography,
)
from compass.ocean.tests.global_ocean.mesh.rrs6to18 import RRS6to18BaseMesh
from compass.ocean.tests.global_ocean.mesh.so12to30 import SO12to30BaseMesh
from compass.ocean.tests.global_ocean.mesh.wc14 import WC14BaseMesh
from compass.ocean.tests.global_ocean.metadata import (
    get_author_and_email_from_git,
)
from compass.testcase import TestCase
from compass.validate import compare_variables


class Mesh(TestCase):
    """
    A test case for creating a global MPAS-Ocean mesh

    Attributes
    ----------
    mesh_name : str
        The name of the mesh

    mesh_subdir : str
        The subdirectory within the test group for all test cases with this
        mesh and topography

    package : str
        The python package for the mesh

    mesh_config_filename : str
        The name of the mesh config file

    with_ice_shelf_cavities : bool
        Whether the mesh includes ice-shelf cavities

    high_res_topography : bool
        Whether to remap a high resolution topography data set.  A lower
        res data set is used for low resolution meshes.

    mali_ais_topo : str
        Short name for the MALI dataset to use for Antarctic Ice Sheet
        topography
    """

    def __init__(self, test_group, mesh_name,  # noqa: C901
                 high_res_topography, mali_ais_topo=None):
        """
        Create test case for creating a global MPAS-Ocean mesh

        Parameters
        ----------
        test_group : compass.ocean.tests.global_ocean.GlobalOcean
            The global ocean test group that this test case belongs to

        mesh_name : str
            The name of the mesh

        high_res_topography : bool
            Whether to remap a high resolution topography data set.  A lower
            res data set is used for low resolution meshes.

        mali_ais_topo : str, optional
            Short name for the MALI dataset to use for Antarctic Ice Sheet
            topography
        """
        name = 'mesh'
        if mali_ais_topo is None:
            self.mesh_subdir = mesh_name
        else:
            self.mesh_subdir = os.path.join(mesh_name,
                                            f'MALI_topo_{mali_ais_topo}')

        subdir = os.path.join(self.mesh_subdir, name)

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
        self.mali_ais_topo = mali_ais_topo
        self.with_ice_shelf_cavities = with_ice_shelf_cavities
        self.high_res_topography = high_res_topography

        name = 'base_mesh'
        subdir = None
        if mesh_name in ['Icos240', 'IcoswISC240']:
            base_mesh_step = IcosahedralMeshStep(
                self, name=name, subdir=subdir, cell_width=240)
        elif mesh_name in ['QU240', 'QUwISC240']:
            base_mesh_step = QuasiUniformSphericalMeshStep(
                self, name=name, subdir=subdir, cell_width=240)
        elif mesh_name in ['Icos', 'IcoswISC']:
            base_mesh_step = IcosMeshFromConfigStep(
                self, name=name, subdir=subdir)
        elif mesh_name in ['QU', 'QUwISC']:
            base_mesh_step = QUMeshFromConfigStep(
                self, name=name, subdir=subdir)
        elif mesh_name in ['EC30to60', 'ECwISC30to60']:
            base_mesh_step = EC30to60BaseMesh(self, name=name, subdir=subdir)
        elif mesh_name in ['ARRM10to60', 'ARRMwISC10to60']:
            base_mesh_step = ARRM10to60BaseMesh(self, name=name, subdir=subdir)
        elif mesh_name in ['RRS6to18', 'RRSwISC6to18']:
            base_mesh_step = RRS6to18BaseMesh(self, name=name, subdir=subdir)
        elif mesh_name in ['SO12to30', 'SOwISC12to30']:
            base_mesh_step = SO12to30BaseMesh(self, name=name, subdir=subdir)
        elif mesh_name in ['FRIS01to60', 'FRISwISC01to60']:
            base_mesh_step = FRIS01to60BaseMesh(self, name=name, subdir=subdir)
        elif mesh_name in ['FRIS02to60', 'FRISwISC02to60']:
            base_mesh_step = FRIS02to60BaseMesh(self, name=name, subdir=subdir)
        elif mesh_name in ['FRIS04to60', 'FRISwISC04to60']:
            base_mesh_step = FRIS04to60BaseMesh(self, name=name, subdir=subdir)
        elif mesh_name in ['FRIS08to60', 'FRISwISC08to60']:
            base_mesh_step = FRIS08to60BaseMesh(self, name=name, subdir=subdir)
        elif mesh_name.startswith('Kuroshio'):
            base_mesh_step = KuroshioBaseMesh(self, name=name, subdir=subdir)
        elif mesh_name in ['WC14', 'WCwISC14']:
            base_mesh_step = WC14BaseMesh(self, name=name, subdir=subdir)
        else:
            raise ValueError(f'Unknown mesh name {mesh_name}')

        self.add_step(base_mesh_step)

        if mali_ais_topo is None:
            remap_step = RemapTopography(test_case=self,
                                         base_mesh_step=base_mesh_step,
                                         mesh_name=mesh_name)
        else:
            remap_step = RemapMaliTopography(
                test_case=self, base_mesh_step=base_mesh_step,
                mesh_name=mesh_name, mali_ais_topo=mali_ais_topo)

        self.add_step(remap_step)

        self.add_step(CullMeshStep(
            test_case=self, base_mesh_step=base_mesh_step,
            with_ice_shelf_cavities=self.with_ice_shelf_cavities,
            remap_topography=remap_step))

    def configure(self, config=None):
        """
        Modify the configuration options for this test case

        config : compass.config.CompassConfigParser, optional
            Configuration options to update if not those for this test case
        """
        if config is None:
            config = self.config
        config.add_from_package('compass.mesh', 'mesh.cfg', exception=True)
        if 'remap_topography' in self.steps:
            config.add_from_package('compass.ocean.mesh',
                                    'remap_topography.cfg', exception=True)

            if not self.high_res_topography:
                config.add_from_package('compass.ocean.mesh',
                                        'low_res_topography.cfg',
                                        exception=True)

        if self.mali_ais_topo is not None:
            package = 'compass.ocean.tests.global_ocean.mesh.' \
                      'remap_mali_topography'
            config.add_from_package(package,
                                    f'{self.mali_ais_topo.lower()}.cfg',
                                    exception=True)

        if self.mesh_name.startswith('Kuroshio'):
            # add the config options for all kuroshio meshes
            config.add_from_package(
                'compass.ocean.tests.global_ocean.mesh.kuroshio',
                'kuroshio.cfg', exception=True)
        config.add_from_package(self.package, self.mesh_config_filename,
                                exception=True)
        if self.mesh_name in ['Icos', 'IcoswISC']:
            # add the config options for all kuroshio meshes
            config.add_from_package(
                'compass.ocean.tests.global_ocean.mesh.qu',
                'icos.cfg', exception=True)

        if self.mesh_name in ['QU', 'QUwISC', 'Icos', 'IcoswISC']:
            res = config.getfloat('global_ocean', 'qu_resolution')
            # roughly area of the ocean divided by the area of a cell
            approx_cell_count = int(4e8 / res**2)
            config.set('global_ocean', 'approx_cell_count',
                       f'{approx_cell_count}')

        config.set('spherical_mesh', 'add_mesh_density', 'True')
        config.set('spherical_mesh', 'plot_cell_width', 'True')
        if self.with_ice_shelf_cavities:
            prefix = config.get('global_ocean', 'prefix')
            config.set('global_ocean', 'prefix', f'{prefix}wISC')
            config.set('global_ocean', 'wisc_description',
                       'Includes cavities under the ice shelves around '
                       'Antarctica')

        # a description of the bathymetry
        if 'remap_topography' in self.steps:
            description = config.get('remap_topography', 'description')
        else:
            description = 'Bathymetry is from GEBCO 2023, combined with ' \
                          'BedMachine Antarctica v3 around Antarctica.'

        config.set('global_ocean', 'bathy_description', description)

        get_author_and_email_from_git(config)

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
