from compass.testgroup import TestGroup

from compass.ocean.tests.global_ocean.mesh import Mesh
from compass.ocean.tests.global_ocean.mesh.qu240.dynamic_adjustment import \
    QU240DynamicAdjustment
from compass.ocean.tests.global_ocean.mesh.ec30to60.dynamic_adjustment import \
    EC30to60DynamicAdjustment
from compass.ocean.tests.global_ocean.mesh.arrm10to60.dynamic_adjustment \
    import ARRM10to60DynamicAdjustment
from compass.ocean.tests.global_ocean.mesh.so12to60.dynamic_adjustment import \
    SO12to60DynamicAdjustment
from compass.ocean.tests.global_ocean.mesh.wc14.dynamic_adjustment import \
    WC14DynamicAdjustment
from compass.ocean.tests.global_ocean.init import Init
from compass.ocean.tests.global_ocean.performance_test import PerformanceTest
from compass.ocean.tests.global_ocean.restart_test import RestartTest
from compass.ocean.tests.global_ocean.decomp_test import DecompTest
from compass.ocean.tests.global_ocean.threads_test import ThreadsTest
from compass.ocean.tests.global_ocean.analysis_test import AnalysisTest
from compass.ocean.tests.global_ocean.daily_output_test import DailyOutputTest
from compass.ocean.tests.global_ocean.monthly_output_test import \
    MonthlyOutputTest
from compass.ocean.tests.global_ocean.files_for_e3sm import FilesForE3SM
from compass.ocean.tests.global_ocean.make_diagnostics_files import \
    MakeDiagnosticsFiles


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
        for mesh_name in ['QU240', 'Icos240', 'QUwISC240']:
            mesh = Mesh(test_group=self, mesh_name=mesh_name)
            self.add_test_case(mesh)

            init = Init(test_group=self, mesh=mesh,
                        initial_condition='PHC',
                        with_bgc=False)
            self.add_test_case(init)

            time_integrator = 'split_explicit'
            self.add_test_case(
                PerformanceTest(
                    test_group=self, mesh=mesh, init=init,
                    time_integrator=time_integrator))
            self.add_test_case(
                RestartTest(
                    test_group=self, mesh=mesh, init=init,
                    time_integrator=time_integrator))
            self.add_test_case(
                DecompTest(
                    test_group=self, mesh=mesh, init=init,
                    time_integrator=time_integrator))
            self.add_test_case(
                ThreadsTest(
                    test_group=self, mesh=mesh, init=init,
                    time_integrator=time_integrator))
            self.add_test_case(
                AnalysisTest(
                    test_group=self, mesh=mesh, init=init,
                    time_integrator=time_integrator))
            self.add_test_case(
                DailyOutputTest(
                    test_group=self, mesh=mesh, init=init,
                    time_integrator=time_integrator))
            self.add_test_case(
                MonthlyOutputTest(
                    test_group=self, mesh=mesh, init=init,
                    time_integrator=time_integrator))

            dynamic_adjustment = QU240DynamicAdjustment(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            self.add_test_case(dynamic_adjustment)
            self.add_test_case(
                FilesForE3SM(
                    test_group=self, mesh=mesh, init=init,
                    dynamic_adjustment=dynamic_adjustment))

            time_integrator = 'RK4'
            self.add_test_case(
                PerformanceTest(
                    test_group=self, mesh=mesh, init=init,
                    time_integrator=time_integrator))
            self.add_test_case(
                RestartTest(
                    test_group=self, mesh=mesh, init=init,
                    time_integrator=time_integrator))
            self.add_test_case(
                DecompTest(
                    test_group=self, mesh=mesh, init=init,
                    time_integrator=time_integrator))
            self.add_test_case(
                ThreadsTest(
                    test_group=self, mesh=mesh, init=init,
                    time_integrator=time_integrator))

            # EN4_1900 tests
            time_integrator = 'split_explicit'
            init = Init(test_group=self, mesh=mesh,
                        initial_condition='EN4_1900',
                        with_bgc=False)
            self.add_test_case(init)
            self.add_test_case(
                PerformanceTest(
                    test_group=self, mesh=mesh, init=init,
                    time_integrator=time_integrator))
            dynamic_adjustment = QU240DynamicAdjustment(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            self.add_test_case(dynamic_adjustment)
            self.add_test_case(
                FilesForE3SM(
                    test_group=self, mesh=mesh, init=init,
                    dynamic_adjustment=dynamic_adjustment))

            # BGC tests
            init = Init(test_group=self, mesh=mesh,
                        initial_condition='PHC',
                        with_bgc=True)
            self.add_test_case(init)
            self.add_test_case(
                PerformanceTest(
                    test_group=self, mesh=mesh, init=init,
                    time_integrator=time_integrator))

        # for other meshes, we do fewer tests
        for mesh_name in ['EC30to60', 'ECwISC30to60']:
            mesh = Mesh(test_group=self, mesh_name=mesh_name)
            self.add_test_case(mesh)

            init = Init(test_group=self, mesh=mesh,
                        initial_condition='PHC',
                        with_bgc=False)
            self.add_test_case(init)

            time_integrator = 'split_explicit'
            self.add_test_case(
                PerformanceTest(
                    test_group=self, mesh=mesh, init=init,
                    time_integrator=time_integrator))
            dynamic_adjustment = EC30to60DynamicAdjustment(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            self.add_test_case(dynamic_adjustment)
            self.add_test_case(
                FilesForE3SM(
                    test_group=self, mesh=mesh, init=init,
                    dynamic_adjustment=dynamic_adjustment))

        # ARRM10to60: just the version without cavities
        for mesh_name in ['ARRM10to60']:
            mesh = Mesh(test_group=self, mesh_name=mesh_name)
            self.add_test_case(mesh)

            init = Init(test_group=self, mesh=mesh,
                        initial_condition='PHC',
                        with_bgc=False)
            self.add_test_case(init)
            time_integrator = 'split_explicit'
            self.add_test_case(
                PerformanceTest(
                    test_group=self, mesh=mesh, init=init,
                    time_integrator=time_integrator))
            dynamic_adjustment = ARRM10to60DynamicAdjustment(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            self.add_test_case(dynamic_adjustment)
            self.add_test_case(
                FilesForE3SM(
                    test_group=self, mesh=mesh, init=init,
                    dynamic_adjustment=dynamic_adjustment))

        # SO12to60: with and without cavities
        for mesh_name in ['SO12to60', 'SOwISC12to60']:
            mesh = Mesh(test_group=self, mesh_name=mesh_name)
            self.add_test_case(mesh)

            init = Init(test_group=self, mesh=mesh,
                        initial_condition='PHC',
                        with_bgc=False)
            self.add_test_case(init)
            time_integrator = 'split_explicit'
            self.add_test_case(
                PerformanceTest(
                    test_group=self, mesh=mesh, init=init,
                    time_integrator=time_integrator))
            dynamic_adjustment = SO12to60DynamicAdjustment(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            self.add_test_case(dynamic_adjustment)
            self.add_test_case(
                FilesForE3SM(
                    test_group=self, mesh=mesh, init=init,
                    dynamic_adjustment=dynamic_adjustment))

        # WC14: just the version without cavities
        for mesh_name in ['WC14']:
            mesh = Mesh(test_group=self, mesh_name=mesh_name)
            self.add_test_case(mesh)

            init = Init(test_group=self, mesh=mesh,
                        initial_condition='PHC',
                        with_bgc=False)
            self.add_test_case(init)
            time_integrator = 'split_explicit'
            self.add_test_case(
                PerformanceTest(
                    test_group=self, mesh=mesh, init=init,
                    time_integrator=time_integrator))
            dynamic_adjustment = WC14DynamicAdjustment(
                test_group=self, mesh=mesh, init=init,
                time_integrator=time_integrator)
            self.add_test_case(dynamic_adjustment)
            self.add_test_case(
                FilesForE3SM(
                    test_group=self, mesh=mesh, init=init,
                    dynamic_adjustment=dynamic_adjustment))

        # A test case for making diagnostics files from an existing mesh
        self.add_test_case(MakeDiagnosticsFiles(test_group=self))
