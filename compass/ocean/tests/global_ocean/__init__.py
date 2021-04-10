from compass.testgroup import TestGroup

from compass.ocean.tests.global_ocean.mesh import Mesh
from compass.ocean.tests.global_ocean.mesh.qu240.dynamic_adjustment import \
    QU240DynamicAdjustment
from compass.ocean.tests.global_ocean.mesh.ec30to60.dynamic_adjustment import \
    EC30to60DynamicAdjustment
from compass.ocean.tests.global_ocean.mesh.so12to60.dynamic_adjustment import \
    SO12to60DynamicAdjustment
from compass.ocean.tests.global_ocean.init import Init
from compass.ocean.tests.global_ocean.performance_test import PerformanceTest
from compass.ocean.tests.global_ocean.restart_test import RestartTest
from compass.ocean.tests.global_ocean.decomp_test import DecompTest
from compass.ocean.tests.global_ocean.threads_test import ThreadsTest
from compass.ocean.tests.global_ocean.analysis_test import AnalysisTest
from compass.ocean.tests.global_ocean.daily_output_test import DailyOutputTest
from compass.ocean.tests.global_ocean.files_for_e3sm import FilesForE3SM


class GlobalOcean(TestGroup):
    """
    A test group for setting up global initial conditions and performing
    regression testing and dynamic adjustment for MPAS-Ocean
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='global_ocean')

        # we do a lot of tests for QU240/QUwISC240
        for mesh_name in ['QU240', 'QUwISC240']:
            mesh = Mesh(test_group=self, mesh_name=mesh_name)

            init = Init(test_group=self, mesh=mesh,
                        initial_condition='PHC',
                        with_bgc=False)

            time_integrator = 'split_explicit'
            PerformanceTest(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            RestartTest(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            DecompTest(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            ThreadsTest(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            AnalysisTest(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            DailyOutputTest(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)

            dynamic_adjustment = QU240DynamicAdjustment(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            FilesForE3SM(
                test_group=self, mesh=mesh, init=init,
                dynamic_adjustment=dynamic_adjustment)

            time_integrator = 'RK4'
            PerformanceTest(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            RestartTest(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            DecompTest(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            ThreadsTest(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)

            # EN4_1900 tests
            time_integrator = 'split_explicit'
            init = Init(test_group=self, mesh=mesh,
                        initial_condition='EN4_1900',
                        with_bgc=False)
            PerformanceTest(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            dynamic_adjustment = QU240DynamicAdjustment(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            FilesForE3SM(
                test_group=self, mesh=mesh, init=init,
                dynamic_adjustment=dynamic_adjustment)

            # BGC tests
            init = Init(test_group=self, mesh=mesh,
                        initial_condition='PHC',
                        with_bgc=True)
            PerformanceTest(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)

        # for other meshes, we do fewer tests
        for mesh_name in ['EC30to60', 'ECwISC30to60']:
            mesh = Mesh(test_group=self, mesh_name=mesh_name)

            init = Init(test_group=self, mesh=mesh,
                        initial_condition='PHC',
                        with_bgc=False)

            time_integrator = 'split_explicit'
            PerformanceTest(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            dynamic_adjustment = EC30to60DynamicAdjustment(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            FilesForE3SM(
                test_group=self, mesh=mesh, init=init,
                dynamic_adjustment=dynamic_adjustment)

        # SOwISC12to60: just the version with cavities for now
        for mesh_name in ['SOwISC12to60']:
            mesh = Mesh(test_group=self, mesh_name=mesh_name)

            init = Init(test_group=self, mesh=mesh,
                        initial_condition='PHC',
                        with_bgc=False)
            time_integrator = 'split_explicit'
            PerformanceTest(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            dynamic_adjustment = SO12to60DynamicAdjustment(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            FilesForE3SM(
                test_group=self, mesh=mesh, init=init,
                dynamic_adjustment=dynamic_adjustment)
