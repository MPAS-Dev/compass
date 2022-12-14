import os

from compass.io import symlink, package_path
from compass.testcase import TestCase
from compass.ocean.tests.global_ocean.files_for_e3sm.ocean_initial_condition \
    import OceanInitialCondition
from compass.ocean.tests.global_ocean.files_for_e3sm.seaice_initial_condition \
    import SeaiceInitialCondition
from compass.ocean.tests.global_ocean.files_for_e3sm.ocean_graph_partition \
    import OceanGraphPartition
from compass.ocean.tests.global_ocean.files_for_e3sm.scrip import Scrip
from compass.ocean.tests.global_ocean.files_for_e3sm.e3sm_to_cmip_maps import \
    E3smToCmipMaps
from compass.ocean.tests.global_ocean.files_for_e3sm.diagnostics_files \
    import DiagnosticsFiles
from compass.ocean.tests.global_ocean.forward import get_forward_subdir
from compass.ocean.tests.global_ocean.configure import configure_global_ocean


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

    restart_filename : str
        A restart file from the end of the dynamic adjustment test case to use
        as the basis for an E3SM initial condition
    """
    def __init__(self, test_group, mesh, init, dynamic_adjustment):
        """
        Create test case for creating a global MPAS-Ocean mesh

        Parameters
        ----------
        test_group : compass.ocean.tests.global_ocean.GlobalOcean
            The global ocean test group that this test case belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that produces the mesh for this run

        init : compass.ocean.tests.global_ocean.init.Init
            The test case that produces the initial condition for this run

        dynamic_adjustment : compass.ocean.tests.global_ocean.dynamic_adjustment.DynamicAdjustment
            The test case that performs dynamic adjustment to dissipate
            fast-moving waves from the initial condition
        """
        name = 'files_for_e3sm'
        time_integrator = dynamic_adjustment.time_integrator
        subdir = get_forward_subdir(init.init_subdir, time_integrator, name)

        super().__init__(test_group=test_group, name=name, subdir=subdir)
        self.mesh = mesh
        self.init = init
        self.dynamic_adjustment = dynamic_adjustment

        restart_filename = os.path.join(
            '..', 'dynamic_adjustment',
            dynamic_adjustment.restart_filenames[-1])
        self.restart_filename = restart_filename

        self.add_step(
            OceanInitialCondition(test_case=self,
                                  restart_filename=restart_filename))

        self.add_step(
            OceanGraphPartition(test_case=self, mesh=mesh,
                                restart_filename=restart_filename))

        self.add_step(
            SeaiceInitialCondition(
                test_case=self, restart_filename=restart_filename,
                with_ice_shelf_cavities=mesh.with_ice_shelf_cavities))

        self.add_step(
            Scrip(
                test_case=self, restart_filename=restart_filename,
                with_ice_shelf_cavities=mesh.with_ice_shelf_cavities))

        self.add_step(
            E3smToCmipMaps(
                test_case=self, restart_filename=restart_filename))

        self.add_step(
            DiagnosticsFiles(
                test_case=self, restart_filename=restart_filename,
                with_ice_shelf_cavities=mesh.with_ice_shelf_cavities))

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        configure_global_ocean(test_case=self, mesh=self.mesh, init=self.init)
        package = 'compass.ocean.tests.global_ocean.files_for_e3sm'
        with package_path(package, 'README') as target:
            symlink(str(target), '{}/README'.format(self.work_dir))
