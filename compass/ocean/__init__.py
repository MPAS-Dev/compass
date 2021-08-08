from compass.mpas_core import MpasCore
from compass.ocean.tests.baroclinic_channel import BaroclinicChannel
from compass.ocean.tests.global_convergence import GlobalConvergence
from compass.ocean.tests.global_ocean import GlobalOcean
from compass.ocean.tests.gotm import Gotm
from compass.ocean.tests.ice_shelf_2d import IceShelf2d
from compass.ocean.tests.isomip_plus import IsomipPlus
from compass.ocean.tests.ziso import Ziso
from compass.ocean.tests.sphere_transport import SphereTransport


class Ocean(MpasCore):
    """
    The collection of all test case for the MPAS-Ocean core
    """

    def __init__(self):
        """
        Construct the collection of MPAS-Ocean test cases
        """
        super().__init__(name='ocean')

        self.add_test_group(BaroclinicChannel(mpas_core=self))
        self.add_test_group(GlobalConvergence(mpas_core=self))
        self.add_test_group(GlobalOcean(mpas_core=self))
        self.add_test_group(Gotm(mpas_core=self))
        self.add_test_group(IceShelf2d(mpas_core=self))
        self.add_test_group(IsomipPlus(mpas_core=self))
        self.add_test_group(Ziso(mpas_core=self))
        self.add_test_group(SphereTransport(mpas_core=self))
