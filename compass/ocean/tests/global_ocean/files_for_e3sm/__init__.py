import os

from compass.io import package_path, symlink
from compass.ocean.tests.global_ocean.configure import configure_global_ocean
from compass.ocean.tests.global_ocean.files_for_e3sm.diagnostic_maps import (
    DiagnosticMaps,
)
from compass.ocean.tests.global_ocean.files_for_e3sm.diagnostic_masks import (
    DiagnosticMasks,
)
from compass.ocean.tests.global_ocean.files_for_e3sm.e3sm_to_cmip_maps import (
    E3smToCmipMaps,
)
from compass.ocean.tests.global_ocean.files_for_e3sm.ocean_graph_partition import (  # noqa: E501
    OceanGraphPartition,
)
from compass.ocean.tests.global_ocean.files_for_e3sm.ocean_initial_condition import (  # noqa: E501
    OceanInitialCondition,
)
from compass.ocean.tests.global_ocean.files_for_e3sm.scrip import Scrip
from compass.ocean.tests.global_ocean.files_for_e3sm.seaice_graph_partition import (  # noqa: E501
    SeaiceGraphPartition,
)
from compass.ocean.tests.global_ocean.files_for_e3sm.seaice_initial_condition import (  # noqa: E501
    SeaiceInitialCondition,
)
from compass.ocean.tests.global_ocean.forward import get_forward_subdir
from compass.testcase import TestCase


class FilesForE3SM(TestCase):
    """
    A test case for assembling files needed for MPAS-Ocean and MPAS-Seaice
    initial conditions in E3SM as well as files needed for diagnostics from
    the Meridional Overturning Circulation analysis member and MPAS-Analysis

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.Mesh
        The test case that produces the mesh for this run

    init : compass.ocean.tests.global_ocean.init.Init
        The test case that produces the initial condition for this run

    dynamic_adjustment : compass.ocean.tests.global_ocean.dynamic_adjustment.DynamicAdjustment
        The test case that performs dynamic adjustment to dissipate
        fast-moving waves from the initial condition
    """  # noqa: E501
    def __init__(self, test_group, mesh=None, init=None,
                 dynamic_adjustment=None):
        """
        Create test case for creating a global MPAS-Ocean mesh

        Parameters
        ----------
        test_group : compass.ocean.tests.global_ocean.GlobalOcean
            The global ocean test group that this test case belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh, optional
            The test case that produces the mesh for this run

        init : compass.ocean.tests.global_ocean.init.Init, optional
            The test case that produces the initial condition for this run

        dynamic_adjustment : compass.ocean.tests.global_ocean.dynamic_adjustment.DynamicAdjustment, optional
            The test case that performs dynamic adjustment to dissipate
            fast-moving waves from the initial condition
        """  # noqa: E501
        name = 'files_for_e3sm'
        if dynamic_adjustment is not None:
            time_integrator = dynamic_adjustment.time_integrator
            subdir = get_forward_subdir(
                init.init_subdir, time_integrator, name)
        else:
            subdir = name

        super().__init__(test_group=test_group, name=name, subdir=subdir)
        self.mesh = mesh
        self.init = init
        self.dynamic_adjustment = dynamic_adjustment

        self.add_step(OceanInitialCondition(test_case=self))
        self.add_step(OceanGraphPartition(test_case=self))
        self.add_step(SeaiceInitialCondition(test_case=self))
        self.add_step(SeaiceGraphPartition(test_case=self))
        self.add_step(Scrip(test_case=self))
        self.add_step(E3smToCmipMaps(test_case=self))
        self.add_step(DiagnosticMaps(test_case=self))
        self.add_step(DiagnosticMasks(test_case=self))

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        mesh = self.mesh
        init = self.init
        dynamic_adjustment = self.dynamic_adjustment
        config = self.config
        work_dir = self.work_dir

        if mesh is not None:
            configure_global_ocean(test_case=self, mesh=mesh,
                                   init=init)
        package = 'compass.ocean.tests.global_ocean.files_for_e3sm'
        with package_path(package, 'README') as target:
            symlink(str(target), f'{work_dir}/README')

        if mesh is not None:
            config.set('files_for_e3sm', 'with_ice_shelf_cavities',
                       f'{mesh.with_ice_shelf_cavities}')

            mesh_path = mesh.get_cull_mesh_path()
            graph_filename = os.path.join(
                self.base_work_dir, mesh_path, 'culled_graph.info')
            graph_filename = os.path.abspath(graph_filename)
            config.set('files_for_e3sm', 'graph_filename', graph_filename)

        if dynamic_adjustment is not None:
            restart_filename = os.path.join(
                work_dir, '..', 'dynamic_adjustment',
                dynamic_adjustment.restart_filenames[-1])
            restart_filename = os.path.abspath(restart_filename)
            config.set('files_for_e3sm', 'ocean_restart_filename',
                       restart_filename)
