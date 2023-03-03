import sys

import numpy as np
from scipy.stats import qmc

from compass.landice.iceshelf_melt import calc_mean_TF
from compass.landice.tests.ensemble_generator.ensemble_manager import (
    EnsembleManager,
)
from compass.landice.tests.ensemble_generator.ensemble_member import (
    EnsembleMember,
)
from compass.testcase import TestCase
from compass.validate import compare_variables


class ThwaitesEnsemble(TestCase):
    """
    A test case for performing an ensemble of Thwaites Glacier
    simulations for uncertainty quantification studies.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ensemble_generator.EnsembleGenerator
            The test group that this test case belongs to

        """
        name = 'thwaites_ensemble'
        super().__init__(test_group=test_group, name=name)

        # We don't want to initialize all the individual runs
        # So during init, we only add the run manager
        self.add_step(EnsembleManager(test_case=self))

    def configure(self):
        """
        Configure a parameter ensemble of a Thwaites Glacier simulations.

        Start by identifying the start and end run numbers to set up
        from the config.

        Next, read a pre-defined unit parameter vector that can be used
        for assigning parameter values to each ensemble member.

        The main work is using the unit parameter vector to set parameter
        values for each parameter to be varied, over prescribed ranges.

        Then create the ensemble member as a step in the test case by calling
        the EnsembleMember constructor.

        Finally, add this step to the test case's step_to_run.  This normally
        happens automatically if steps are added to the test case in the test
        case constructor, but because we waited to add these steps until this
        configure phase, we must explicitly add the steps to steps_to_run.
        """

        # Define some constants
        rhoi = 910.0
        rhosw = 1028.0
        cp_seawater = 3.974e3
        latent_heat_ice = 335.0e3
        sec_in_yr = 3600.0 * 24.0 * 365.0

        # Determine start and end run numbers being requested
        self.start_run = self.config.getint('ensemble', 'start_run')
        self.end_run = self.config.getint('ensemble', 'end_run')

        # Define parameters being sampled and their ranges

        # Determine how many and which parameters are being used
        use_fric_exp = self.config.getboolean('ensemble', 'use_fric_exp')
        use_mu_scale = self.config.getboolean('ensemble', 'use_mu_scale')
        use_stiff_scale = self.config.getboolean('ensemble',
                                                 'use_stiff_scale')
        use_sigma_max = self.config.getboolean('ensemble', 'use_sigma_max')
        use_calv_limit = self.config.getboolean('ensemble', 'use_calv_limit')
        use_gamma0 = self.config.getboolean('ensemble', 'use_gamma0')
        use_meltflux = self.config.getboolean('ensemble', 'use_meltflux')

        n_params = (use_fric_exp + use_mu_scale + use_stiff_scale +
                    use_sigma_max + use_calv_limit + use_gamma0 +
                    use_meltflux)
        if n_params == 0:
            sys.exit("ERROR: At least one parameter must be specified.")

        # Generate unit Sobol sequence for number of parameters being used
        print(f"Generating Sobol sequence for {n_params} parameters")
        sampler = qmc.Sobol(d=n_params, scramble=True, seed=4)
        max_samples = 512  # Can make this bigger than ever needed
        param_unit_values = sampler.random(n=max_samples)

        # Now define parameter ranges for each param being used
        idx = 0

        # basal fric exp
        if use_fric_exp:
            print('Including basal friction exponent')
            minval = self.config.getfloat('ensemble', 'fric_exp_min')
            maxval = self.config.getfloat('ensemble', 'fric_exp_max')
            basal_fric_exp_vec = param_unit_values[:, idx] * \
                (maxval - minval) + minval
            idx += 1
        else:
            basal_fric_exp_vec = [None] * max_samples

        # mu scale
        if use_mu_scale:
            print('Including scaling of muFriction')
            minval = self.config.getfloat('ensemble', 'mu_scale_min')
            maxval = self.config.getfloat('ensemble', 'mu_scale_max')
            mu_scale_vec = param_unit_values[:, idx] * \
                (maxval - minval) + minval
            idx += 1
        else:
            mu_scale_vec = [None] * max_samples

        # stiff scale
        if use_stiff_scale:
            print('Including scaling of stiffnessFactor')
            minval = self.config.getfloat('ensemble', 'stiff_scale_min')
            maxval = self.config.getfloat('ensemble', 'stiff_scale_max')
            stiff_scale_vec = param_unit_values[:, idx] * \
                (maxval - minval) + minval
            idx += 1
        else:
            stiff_scale_vec = [None] * max_samples

        # von mises threshold stress
        if use_sigma_max:
            print('Including sigma_max')
            minval = self.config.getfloat('ensemble', 'sigma_max_min')
            maxval = self.config.getfloat('ensemble', 'sigma_max_max')
            von_mises_threshold_vec = param_unit_values[:, idx] * \
                (maxval - minval) + minval
            idx += 1
        else:
            von_mises_threshold_vec = [None] * max_samples

        # calving speed limit
        if use_calv_limit:
            print('Including calving speed limit')
            minval = self.config.getfloat('ensemble', 'calv_limit_min')
            maxval = self.config.getfloat('ensemble', 'calv_limit_max')
            calv_spd_lim_vec = param_unit_values[:, idx] * \
                (maxval - minval) + minval
            calv_spd_lim_vec /= sec_in_yr  # convert from m/yr to s/yr
            idx += 1
        else:
            calv_spd_lim_vec = [None] * max_samples

        # gamma0
        if use_gamma0:
            print('Including gamma0')
            # gamma0
            minval = self.config.getfloat('ensemble', 'gamma0_min')
            maxval = self.config.getfloat('ensemble', 'gamma0_max')
            gamma0_vec = param_unit_values[:, idx] * \
                (maxval - minval) + minval
            idx += 1
        else:
            gamma0_vec = [None] * max_samples

        # melt flux
        if use_meltflux:
            # melt flux
            minval = self.config.getfloat('ensemble', 'meltflux_min')
            maxval = self.config.getfloat('ensemble', 'meltflux_max')
            meltflux_vec = param_unit_values[:, idx] * \
                (maxval - minval) + minval
            idx += 1
            iceshelf_area_obs = self.config.getfloat('ensemble',
                                                     'iceshelf_area_obs')

            # deltaT
            section = self.config['ensemble']
            input_file_path = section.get('input_file_path')
            TF_file_path = section.get('TF_file_path')
            mean_TF, iceshelf_area = calc_mean_TF(input_file_path,
                                                  TF_file_path)
            # Adjust observed melt flux for ice-shelf area in init. condition
            print(f'IS area: model={iceshelf_area}, Obs={iceshelf_area_obs}')
            area_correction = iceshelf_area / iceshelf_area_obs
            print(f"Ice-shelf area correction is {area_correction}.")
            if (np.absolute(area_correction - 1.0) > 0.2):
                sys.exit("ERROR: ice-shelf area correction is larger than "
                         "20%. Check data consistency before proceeding.")
            meltflux_vec *= iceshelf_area / iceshelf_area_obs
            TFs = np.linspace(-5.0, 10.0, num=int(15.0 / 0.01))
            c_melt = (rhosw * cp_seawater / (rhoi * latent_heat_ice))**2
            deltaT_vec = np.zeros(max_samples)
            for ii in range(self.start_run, self.end_run + 1):
                meltfluxes = (gamma0_vec[ii] * c_melt * TFs *
                              np.absolute(TFs) *
                              iceshelf_area) * rhoi / 1.0e12  # Gt/yr
                # interpolate deltaT value.  Use nan values outside of range
                # so out of range results get detected
                deltaT_vec[ii] = np.interp(meltflux_vec[ii], meltfluxes, TFs,
                                           left=np.nan,
                                           right=np.nan) - mean_TF
                if np.isnan(deltaT_vec[ii]):
                    sys.exit("ERROR: interpolated deltaT out of range. "
                             "Adjust definition of 'TFs'")
        else:
            deltaT_vec = [None] * max_samples

        # add runs as steps based on the run range requested
        if self.end_run > max_samples:
            sys.exit("Error: end_run specified in config exceeds maximum "
                     "sample size available in param_vector_filename")
        for run_num in range(self.start_run, self.end_run + 1):
            self.add_step(EnsembleMember(test_case=self, run_num=run_num,
                          test_resources_location='compass.landice.tests.ensemble_generator.thwaites',  # noqa
                          basal_fric_exp=basal_fric_exp_vec[run_num],
                          mu_scale=mu_scale_vec[run_num],
                          stiff_scale=stiff_scale_vec[run_num],
                          von_mises_threshold=von_mises_threshold_vec[run_num],
                          calv_spd_lim=calv_spd_lim_vec[run_num],
                          gamma0=gamma0_vec[run_num],
                          deltaT=deltaT_vec[run_num]))
            # Note: do not add to steps_to_run, because ensemble_manager
            # will handle submitting and running the runs

        # Have compass run only run the run_manager but not any actual runs.
        # This is because the individual runs will be submitted as jobs
        # by the ensemble manager.
        self.steps_to_run = ['ensemble_manager',]

    # no run() method is needed

    # no validate() method is needed
