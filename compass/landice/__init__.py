from compass.mpas_core import MpasCore
from compass.landice.tests.antarctica import Antarctica
from compass.landice.tests.calving_dt_convergence import CalvingDtConvergence
from compass.landice.tests.circular_shelf import CircularShelf
from compass.landice.tests.dome import Dome
from compass.landice.tests.eismint2 import Eismint2
from compass.landice.tests.ensemble_generator import EnsembleGenerator
from compass.landice.tests.enthalpy_benchmark import EnthalpyBenchmark
from compass.landice.tests.greenland import Greenland
from compass.landice.tests.humboldt import Humboldt
from compass.landice.tests.hydro_radial import HydroRadial
from compass.landice.tests.ismip6_forcing import Ismip6Forcing
from compass.landice.tests.kangerlussuaq import Kangerlussuaq
from compass.landice.tests.koge_bugt_s import KogeBugtS
from compass.landice.tests.mismipplus import MISMIPplus
from compass.landice.tests.thwaites import Thwaites


class Landice(MpasCore):
    """
    The collection of all test case for the MALI core
    """

    def __init__(self):
        """
        Construct the collection of MALI test cases
        """
        super().__init__(name='landice')

        self.add_test_group(Antarctica(mpas_core=self))
        self.add_test_group(CalvingDtConvergence(mpas_core=self))
        self.add_test_group(CircularShelf(mpas_core=self))
        self.add_test_group(Dome(mpas_core=self))
        self.add_test_group(Eismint2(mpas_core=self))
        self.add_test_group(EnsembleGenerator(mpas_core=self))
        self.add_test_group(EnthalpyBenchmark(mpas_core=self))
        self.add_test_group(Greenland(mpas_core=self))
        self.add_test_group(Humboldt(mpas_core=self))
        self.add_test_group(HydroRadial(mpas_core=self))
        self.add_test_group(Ismip6Forcing(mpas_core=self))
        self.add_test_group(Kangerlussuaq(mpas_core=self))
        self.add_test_group(KogeBugtS(mpas_core=self))
        self.add_test_group(MISMIPplus(mpas_core=self))
        self.add_test_group(Thwaites(mpas_core=self))
