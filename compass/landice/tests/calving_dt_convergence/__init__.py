from compass.testgroup import TestGroup
from compass.landice.tests.calving_dt_convergence.dt_convergence_test \
     import DtConvergenceTest


class CalvingDtConvergence(TestGroup):
    """
    A test group for a series of test cases for assessing timestep
    convergence related to calving.
    This test group uses a pre-made mesh files and various calving
    and velocity options.
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='calving_dt_convergence')

        for mesh in [
                     'mismip+',
                     'humboldt',
                     'thwaites'
                    ]:
            for calving in [
                            'specified_calving_velocity',
                            'von_Mises_stress',
                            'eigencalving'
                           ]:
                for velo in [
                             'none',
                             'FO'
                            ]:
                    if (calving == 'specified_calving_velocity' and
                       velo == 'FO'):
                        continue  # This combination is not useful
                    self.add_test_case(DtConvergenceTest(test_group=self,
                                                         mesh=mesh,
                                                         calving=calving,
                                                         velo=velo))
