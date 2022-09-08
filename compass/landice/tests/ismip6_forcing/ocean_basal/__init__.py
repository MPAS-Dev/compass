from compass.testcase import TestCase
from compass.landice.tests.ismip6_forcing.ocean_basal.process_basal_melt \
    import ProcessBasalMelt


class OceanBasal(TestCase):
    """
    A test case for processing the ISMIP6 ocean basalmelt param. coeff. data.
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
        name = 'ocean_basal'
        super().__init__(test_group=test_group, name=name)

        step = ProcessBasalMelt(test_case=self)
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

    # input files: input uniform melt rate coefficient (gamma0)
    # and temperature correction per basin
    _files = {
        "2100":["coeff_gamma0_DeltaT_quadratic_local_5th_pct_PIGL_gamma_calibration.nc",
                "coeff_gamma0_DeltaT_quadratic_local_5th_percentile.nc",
                "coeff_gamma0_DeltaT_quadratic_local_95th_pct_PIGL_gamma_calibration.nc",
                "coeff_gamma0_DeltaT_quadratic_local_95th_percentile.nc",
                "coeff_gamma0_DeltaT_quadratic_local_median_PIGL_gamma_calibration.nc",
                "coeff_gamma0_DeltaT_quadratic_local_median.nc",
                "coeff_gamma0_DeltaT_quadratic_non_local_5th_pct_PIGL_gamma_calibration.nc",
                "coeff_gamma0_DeltaT_quadratic_non_local_5th_percentile.nc",
                "coeff_gamma0_DeltaT_quadratic_non_local_95th_pct_PIGL_gamma_calibration.nc",
                "coeff_gamma0_DeltaT_quadratic_non_local_95th_percentile.nc",
                "coeff_gamma0_DeltaT_quadratic_non_local_median_PIGL_gamma_calibration.nc",
                "coeff_gamma0_DeltaT_quadratic_non_local_median.nc"]
    }