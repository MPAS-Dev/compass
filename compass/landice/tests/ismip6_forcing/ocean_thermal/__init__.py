from compass.testcase import TestCase
from compass.landice.tests.ismip6_forcing.ocean_thermal.process_thermal_forcing\
    import ProcessThermalForcing


class OceanThermal(TestCase):
    """
    A test case for processing the ISMIP6 ocean thermal forcing data.
    The test case builds a mapping file for interpolation between the
    ISMIP6 8km polarstereo grid and MALI mesh, regrids the forcing data
    and renames the ISMIP6 variables to corresponding MALI variables.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ismip6_forcing.Ismip6Forcing
            The test group that this test case belongs to
        """
        name = 'ocean_thermal'
        super().__init__(test_group=test_group, name=name)

        step = ProcessThermalForcing(test_case=self)
        self.add_step(step)

    def configure(self):
        """
        Configures test case
        """
        base_path = self.config.get(section="ismip6_ais",
                                     option="base_path")
        model = self.config.get(section="ismip6_ais",
                                     option="model")
        scenario = self.config.get(section="ismip6_ais",
                                     option="scenario")
        period_endyear = self.config.get(section="ismip6_ais",
                                     option="period_endyear")
        mali_mesh_name = self.config.get(section="ismip6_ais",
                                     option="mali_mesh_name")
        mali_mesh_file = self.config.get(section="ismip6_ais",
                                     option="mali_mesh_file")

        if base_path == "NotAvailable":
            raise ValueError("You need to supply a user config file, which "
                             "should contain the ismip6_ais "
                             "section with the base_path option")
        if model == "NotAvailable":
            raise ValueError("You need to supply a user config file, which "
                             "should contain the ismip6_ais "
                             "section with the model option")
        if scenario == "NotAvailable":
            raise ValueError("You need to supply a user config file, which "
                             "should contain the ismip6_ais "
                             "section with the scenario option")
        if period_endyear == "NotAvailable":
            raise ValueError("You need to supply a user config file, which "
                             "should contain the ismip6_ais "
                             "section with the period_endyear option")
        if mali_mesh_name == "NotAvailable":
            raise ValueError("You need to supply a user config file, which "
                             "should contain the ismip6_ais "
                             "section with the mali_mesh_name option")
        if mali_mesh_file == "NotAvailable":
            raise ValueError("You need to supply a user config file, which "
                             "should contain the ismip6_ais "
                             "section with the mali_mesh_file option")

    _files = {
        "2100": {
            "HadGEM2-ES":{
                "RCP85": ["file1", "file2"],
                "RCP26": ["file1", "file2"]
            },
            "CESM2-WACCM":{}
        },
        "2300": {
            "CCSM4":{
                "RCP85":["AIS/Ocean_forcing/ccsm4_RCP85/1995-2300/CCSM4_RCP85_thermal_forcing_8km_x_60m.nc"]
            },
            "CESM2-WACCM":{
                "SSP585":["AIS/Ocean_forcing/CESM2-WACCM_ssp585/1995-2299/CESM2-WACCM_SSP585_thermal_forcing_8km_x_60m.nc"] #repeat: "1AIS/Ocean_forcing/cesm2-waccm_ssp585-repeat/1995-2300/CESM2-WACCM_SSP585_thermal_forcing_8km_x_60m.nc"
            },
           # "CSIRO-Mk3-6-0" does not exist for ocean forcing
            "HadGEM2-ES":{
                "RCP85":["AIS/Ocean_forcing/hadgem2-es_RCP85/1995-2299/HadGEM2-ES_RCP85_thermal_forcing_8km_x_60m.nc",
                         "AIS/Ocean_forcing/hadgem2-es_RCP85-repeat/1995-2299/HadGEM2-ES_RCP85_thermal_forcing_8km_x_60m.nc"]
            },
            "NorESM1-M":{
                "RCP26":["AIS/Ocean_forcing/noresm1-m_RCP26-repeat/1995-2300/NorESM1-M_RCP26_thermal_forcing_8km_x_60m.nc"], #this is repeat forcing
                "RCP85":["/AIS/Ocean_forcing/noresm1-m_RCP85-repeat/1995-2300/NorESM1-M_thermal_forcing_8km_x_60m.nc"]
            },
            "UKESM1-0-LL":{
                "SSP126":["AIS/Ocean_forcing/ukesm1-0-ll_ssp126/1995-2300/UKESM1-0-LL_thermal_forcing_8km_x_60m.nc"],
                "SSP585":["AIS/Ocean_forcing/ukesm1-0-ll_ssp585/1995-2300/UKESM1-0-LL_SSP585_thermal_forcing_8km_x_60m.nc",
                          "AIS/Ocean_forcing/ukesm1-0-ll_ssp585-repeat/1995-2300/UKESM1-0-LL_SSP585_thermal_forcing_8km_x_60m.nc"] #this is repeat forcing again
            }
        }
    }