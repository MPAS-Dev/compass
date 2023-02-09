from compass.mpas_core import MpasCore
from compass.ocean.tests.baroclinic_channel import BaroclinicChannel
from compass.ocean.tests.dam_break import DamBreak
from compass.ocean.tests.drying_slope import DryingSlope
from compass.ocean.tests.global_convergence import GlobalConvergence
from compass.ocean.tests.global_ocean import GlobalOcean
from compass.ocean.tests.gotm import Gotm
from compass.ocean.tests.hurricane import Hurricane
from compass.ocean.tests.internal_wave import InternalWave
from compass.ocean.tests.ice_shelf_2d import IceShelf2d
from compass.ocean.tests.isomip_plus import IsomipPlus
from compass.ocean.tests.merry_go_round import MerryGoRound
from compass.ocean.tests.nonhydro import Nonhydro
from compass.ocean.tests.overflow import Overflow
from compass.ocean.tests.planar_convergence import PlanarConvergence
from compass.ocean.tests.soma import Soma
from compass.ocean.tests.sphere_transport import SphereTransport
from compass.ocean.tests.spherical_harmonic_transform import \
    SphericalHarmonicTransform
from compass.ocean.tests.tides import Tides
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

        self.add_test_group(BaroclinicChannel(mpas_core=self))
        self.add_test_group(DamBreak(mpas_core=self))
        self.add_test_group(DryingSlope(mpas_core=self))
        self.add_test_group(GlobalConvergence(mpas_core=self))
        self.add_test_group(GlobalOcean(mpas_core=self))
        self.add_test_group(Gotm(mpas_core=self))
        self.add_test_group(Hurricane(mpas_core=self))
        self.add_test_group(InternalWave(mpas_core=self))
        self.add_test_group(IceShelf2d(mpas_core=self))
        self.add_test_group(IsomipPlus(mpas_core=self))
        self.add_test_group(MerryGoRound(mpas_core=self))
        self.add_test_group(Nonhydro(mpas_core=self))
        self.add_test_group(Overflow(mpas_core=self))
        self.add_test_group(PlanarConvergence(mpas_core=self))
        self.add_test_group(Soma(mpas_core=self))
        self.add_test_group(SphereTransport(mpas_core=self))
        self.add_test_group(SphericalHarmonicTransform(mpas_core=self))
        self.add_test_group(Tides(mpas_core=self))
        self.add_test_group(Ziso(mpas_core=self))
