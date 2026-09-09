from compass.landice.tests.ismip7_calibration.replication import Replication
from compass.testgroup import TestGroup


class Ismip7Calibration(TestGroup):
    """
    A test group for calibrating MALI's sub-shelf melt parameterization
    against the ISMIP7 Antarctic ice-ocean protocol (Reese et al., Sect. 4.2)

    The protocol asks each ice-sheet model to calibrate the free parameter of
    its melt module against four objective-function terms -- basin-integrated
    present-day melt, melt by buttressing bin, the warm-minus-cold sensitivity
    of ocean models, and observed Amundsen ice-shelf melt -- and to report the
    5th, 50th and 95th percentiles of the resulting parameter distribution.
    """

    def __init__(self, mpas_core):
        """
        Create the test group

        Parameters
        ----------
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='ismip7_calibration')

        self.add_test_case(Replication(test_group=self))
