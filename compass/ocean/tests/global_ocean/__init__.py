from compass.ocean.tests.global_ocean.analysis_test import AnalysisTest
from compass.ocean.tests.global_ocean.daily_output_test import DailyOutputTest
from compass.ocean.tests.global_ocean.decomp_test import DecompTest
from compass.ocean.tests.global_ocean.dynamic_adjustment import (
    DynamicAdjustment,
)
from compass.ocean.tests.global_ocean.files_for_e3sm import FilesForE3SM
from compass.ocean.tests.global_ocean.init import Init
from compass.ocean.tests.global_ocean.mesh import Mesh
from compass.ocean.tests.global_ocean.monthly_output_test import (
    MonthlyOutputTest,
)
from compass.ocean.tests.global_ocean.performance_test import PerformanceTest
from compass.ocean.tests.global_ocean.restart_test import RestartTest
from compass.ocean.tests.global_ocean.threads_test import ThreadsTest
from compass.testgroup import TestGroup


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
        self._add_tests(mesh_names=['QU240', 'Icos240', 'QUwISC240'],
                        high_res_topography=False,
                        include_rk4=True,
                        include_regression=True,
                        include_phc=True,
                        include_en4_1900=True)

        # for other meshes, we do fewer tests
        self._add_tests(mesh_names=['QU', 'Icos', 'QUwISC', 'IcoswISC'])

        self._add_tests(mesh_names=['EC30to60', 'ECwISC30to60'])

        self._add_tests(mesh_names=['ARRM10to60', 'ARRMwISC10to60'])

        self._add_tests(mesh_names=['SO12to60', 'SOwISC12to60'])

        self._add_tests(mesh_names=['WC14', 'WCwISC14'])

        # RRS6to18: with and without cavities
        self._add_tests(mesh_names=['RRS6to18', 'RRSwISC6to18'])

        # Kuroshio meshes without ice-shelf cavities
        self._add_tests(mesh_names=['Kuroshio12to60', 'Kuroshio8to60'])

        # A test case for making E3SM support files from an existing mesh
        self.add_test_case(FilesForE3SM(test_group=self))

    def _add_tests(self, mesh_names, high_res_topography=True,
                   include_rk4=False,
                   include_regression=False, include_phc=False,
                   include_en4_1900=False):
        """ Add test cases for the given mesh(es) """

        default_ic = 'WOA23'
        default_time_int = 'split_explicit_ab2'

        for mesh_name in mesh_names:
            mesh_test = Mesh(test_group=self, mesh_name=mesh_name,
                             high_res_topography=high_res_topography)
            self.add_test_case(mesh_test)

            init_test = Init(test_group=self, mesh=mesh_test,
                             initial_condition=default_ic)
            self.add_test_case(init_test)

            time_integrator = default_time_int
            self.add_test_case(
                PerformanceTest(
                    test_group=self, mesh=mesh_test, init=init_test,
                    time_integrator=time_integrator))
            if include_regression:
                self.add_test_case(
                    RestartTest(
                        test_group=self, mesh=mesh_test, init=init_test,
                        time_integrator=time_integrator))
                self.add_test_case(
                    DecompTest(
                        test_group=self, mesh=mesh_test, init=init_test,
                        time_integrator=time_integrator))
                self.add_test_case(
                    ThreadsTest(
                        test_group=self, mesh=mesh_test, init=init_test,
                        time_integrator=time_integrator))
                self.add_test_case(
                    AnalysisTest(
                        test_group=self, mesh=mesh_test, init=init_test,
                        time_integrator=time_integrator))
                self.add_test_case(
                    DailyOutputTest(
                        test_group=self, mesh=mesh_test, init=init_test,
                        time_integrator=time_integrator))
                self.add_test_case(
                    MonthlyOutputTest(
                        test_group=self, mesh=mesh_test, init=init_test,
                        time_integrator=time_integrator))

            dynamic_adjustment_test = DynamicAdjustment(
                test_group=self, mesh=mesh_test, init=init_test,
                time_integrator=time_integrator)
            self.add_test_case(dynamic_adjustment_test)

            self.add_test_case(
                FilesForE3SM(
                    test_group=self, mesh=mesh_test, init=init_test,
                    dynamic_adjustment=dynamic_adjustment_test))

            if include_rk4:
                time_integrator = 'RK4'
                self.add_test_case(
                    PerformanceTest(
                        test_group=self, mesh=mesh_test, init=init_test,
                        time_integrator=time_integrator))
                self.add_test_case(
                    RestartTest(
                        test_group=self, mesh=mesh_test, init=init_test,
                        time_integrator=time_integrator))
                self.add_test_case(
                    DecompTest(
                        test_group=self, mesh=mesh_test, init=init_test,
                        time_integrator=time_integrator))
                self.add_test_case(
                    ThreadsTest(
                        test_group=self, mesh=mesh_test, init=init_test,
                        time_integrator=time_integrator))

            initial_conditions = []
            if include_phc:
                initial_conditions.append('PHC')
            if include_en4_1900:
                initial_conditions.append('EN4_1900')

            for initial_condition in initial_conditions:
                # additional initial conditions (if any)
                time_integrator = default_time_int
                init_test = Init(test_group=self, mesh=mesh_test,
                                 initial_condition=initial_condition)
                self.add_test_case(init_test)

                self.add_test_case(
                    PerformanceTest(
                        test_group=self, mesh=mesh_test, init=init_test,
                        time_integrator=time_integrator))

                dynamic_adjustment_test = DynamicAdjustment(
                    test_group=self, mesh=mesh_test, init=init_test,
                    time_integrator=time_integrator)
                self.add_test_case(dynamic_adjustment_test)

                self.add_test_case(
                    FilesForE3SM(
                        test_group=self, mesh=mesh_test, init=init_test,
                        dynamic_adjustment=dynamic_adjustment_test))
