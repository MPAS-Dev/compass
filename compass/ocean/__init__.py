from compass.mpas_core import MpasCore
from compass.ocean.tests.baroclinic_channel import BaroclinicChannel
from compass.ocean.tests.global_ocean import GlobalOcean
from compass.ocean.tests.ice_shelf_2d import IceShelf2d
from compass.ocean.tests.ziso import Ziso


class Ocean(MpasCore):
    """
    The collection of all test case for the MPAS-Ocean core
    """

    def __init__(self):
        """
        Construct the collection of MPAS-Ocean test cases
        """
        super().__init__(name='ocean')

        BaroclinicChannel(mpas_core=self)
        GlobalOcean(mpas_core=self)
        IceShelf2d(mpas_core=self)
        Ziso(mpas_core=self)
