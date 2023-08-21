import os

from compass.io import package_path, symlink
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
from compass.ocean.tests.global_ocean.files_for_e3sm.ocean_mesh import (  # noqa: E501
    OceanMesh,
)
from compass.ocean.tests.global_ocean.files_for_e3sm.remap_ice_shelf_melt import (  # noqa: E501
    RemapIceShelfMelt,
)
from compass.ocean.tests.global_ocean.files_for_e3sm.remap_iceberg_climatology import (  # noqa: E501
    RemapIcebergClimatology,
)
from compass.ocean.tests.global_ocean.files_for_e3sm.remap_sea_surface_salinity_restoring import (  # noqa: E501
    RemapSeaSurfaceSalinityRestoring,
)
from compass.ocean.tests.global_ocean.files_for_e3sm.scrip import Scrip
from compass.ocean.tests.global_ocean.files_for_e3sm.seaice_graph_partition import (  # noqa: E501
    SeaiceGraphPartition,
)
from compass.ocean.tests.global_ocean.files_for_e3sm.seaice_initial_condition import (  # noqa: E501
    SeaiceInitialCondition,
)
from compass.ocean.tests.global_ocean.files_for_e3sm.seaice_mesh import (  # noqa: E501
    SeaiceMesh,
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

    data_ice_shelf_melt : compass.ocean.tests.global_ocean.data_ice_shelf_melt.DataIceShelfMelt
        A test case for remapping observed melt rates to the MPAS grid
    """  # noqa: E501
    def __init__(self, test_group, mesh=None, init=None,
                 dynamic_adjustment=None, data_ice_shelf_melt=None):
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

        data_ice_shelf_melt : compass.ocean.tests.global_ocean.data_ice_shelf_melt.DataIceShelfMelt, optional
            A test case for remapping observed melt rates to the MPAS grid
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
        self.data_ice_shelf_melt = data_ice_shelf_melt

        # add metadata if we're running this on an existing mesh
        add_metadata = (dynamic_adjustment is None)
        self.add_step(OceanMesh(test_case=self))
        self.add_step(OceanInitialCondition(test_case=self,
                                            add_metadata=add_metadata))
        self.add_step(OceanGraphPartition(test_case=self))
        self.add_step(SeaiceMesh(test_case=self))
        self.add_step(SeaiceInitialCondition(test_case=self))
        self.add_step(SeaiceGraphPartition(test_case=self))
        self.add_step(Scrip(test_case=self))
        self.add_step(E3smToCmipMaps(test_case=self))
        self.add_step(DiagnosticMaps(test_case=self))
        self.add_step(DiagnosticMasks(test_case=self))

        self.add_step(RemapIceShelfMelt(
            test_case=self,
            data_ice_shelf_melt=data_ice_shelf_melt))

        self.add_step(RemapSeaSurfaceSalinityRestoring(
            test_case=self))

        self.add_step(RemapIcebergClimatology(
            test_case=self))

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        init = self.init
        if init is not None:
            init.configure(config=self.config)

        mesh = self.mesh
        dynamic_adjustment = self.dynamic_adjustment
        config = self.config
        work_dir = self.work_dir

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

        if init is not None:
            if mesh.with_ice_shelf_cavities:
                initial_state_filename = \
                    f'{init.path}/ssh_adjustment/adjusted_init.nc'
            else:
                initial_state_filename = \
                    f'{init.path}/initial_state/initial_state.nc'
            initial_state_filename = os.path.join(self.base_work_dir,
                                                  initial_state_filename)
            config.set('files_for_e3sm', 'ocean_initial_state_filename',
                       initial_state_filename)

        if dynamic_adjustment is not None:
            restart_filename = os.path.join(
                work_dir, '..', 'dynamic_adjustment',
                dynamic_adjustment.restart_filenames[-1])
            restart_filename = os.path.abspath(restart_filename)
            config.set('files_for_e3sm', 'ocean_restart_filename',
                       restart_filename)
