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


class ControlEnsemble(TestCase):
    """
    A test case for performing an ensemble of
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
        name = 'control_ensemble'
        super().__init__(test_group=test_group, name=name)

        # We don't want to initialize all the individual runs
        # So during init, we only add the run manager
        self.add_step(EnsembleManager(test_case=self))

    def configure(self):
        """
        Configure a parameter ensemble of MALI simulations.

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
        c_melt = (rhosw * cp_seawater / (rhoi * latent_heat_ice))**2

        section = self.config['ensemble']

        # Determine start and end run numbers being requested
        self.start_run = section.getint('start_run')
        self.end_run = section.getint('end_run')

        # Define parameters being sampled and their ranges
        param_list = ['fric_exp', 'mu_scale', 'stiff_scale',
                      'von_mises_threshold', 'calv_limit', 'gamma0',
                      'meltflux']

        # Determine how many and which parameters are being used
        n_params = 0
        param_dict = {}
        for param in param_list:
            param_dict[param] = {}
            param_dict[param]['active'] = section.getboolean(f'use_{param}')
            n_params += param_dict[param]['active']
        if n_params == 0:
            sys.exit("ERROR: At least one parameter must be specified.")

        # Generate unit parameter vectors - either uniform or Sobol
        sampling_method = section.get('sampling_method')
        max_samples = section.getint('max_samples')
        if max_samples < self.end_run:
            sys.exit("ERROR: max_samples is exceeded by end_run")
        if sampling_method == 'sobol':
            # Generate unit Sobol sequence for number of parameters being used
            print(f"Generating Sobol sequence for {n_params} parameter(s)")
            sampler = qmc.Sobol(d=n_params, scramble=True, seed=4)
            param_unit_values = sampler.random(n=max_samples)
        elif sampling_method == 'uniform':
            print(f"Generating uniform sampling for {n_params} parameter(s)")
            samples = np.linspace(0.0, 1.0, max_samples).reshape(-1, 1)
            param_unit_values = np.tile(samples, (1, n_params))
        else:
            sys.exit("ERROR: Unsupported sampling method specified.")

        # Define parameter vectors for each param being used
        idx = 0
        for param in param_list:
            if param_dict[param]['active']:
                print('Including parameter ' + param)
                min_val = section.getfloat(f'{param}_min')
                max_val = section.getfloat(f'{param}_max')
                param_dict[param]['vec'] = param_unit_values[:, idx] * \
                    (max_val - min_val) + min_val
                idx += 1
            else:
                param_dict[param]['vec'] = np.full((max_samples,), None)

        # Deal with a few special cases

        # change units on calving speed limit from m/yr to s/yr
        if param_dict['calv_limit']['active']:
            param_dict['calv_limit']['vec'] = \
                param_dict['calv_limit']['vec'][:] / sec_in_yr

        # melt flux needs to be converted to deltaT
        if param_dict['meltflux']['active']:
            # First calculate mean TF for this domain
            iceshelf_area_obs = section.getfloat('iceshelf_area_obs')
            input_file_path = section.get('input_file_path')
            TF_file_path = section.get('TF_file_path')
            mean_TF, iceshelf_area = calc_mean_TF(input_file_path,
                                                  TF_file_path)

            # Adjust observed melt flux for ice-shelf area in init. condition
            print(f'IS area: model={iceshelf_area}, Obs={iceshelf_area_obs}')
            area_correction = iceshelf_area / iceshelf_area_obs
            print(f"Ice-shelf area correction is {area_correction}.")
            if (np.absolute(area_correction - 1.0) > 0.2):
                print("WARNING: ice-shelf area correction is larger than "
                      "20%. Check data consistency before proceeding.")
            param_dict['meltflux']['vec'] *= iceshelf_area / iceshelf_area_obs

            # Set up an array of TF values to use for linear interpolation
            # Make it span a large enough range to capture deltaT what would
            # be needed for the range of gamma0 values considered.
            # Not possible to know a priori, so pick a wide range.
            TFs = np.linspace(-5.0, 10.0, num=int(15.0 / 0.01))
            deltaT_vec = np.zeros(max_samples)
            # For each run, calculate the deltaT needed to obtain the target
            # melt flux
            for ii in range(self.start_run, self.end_run + 1):
                # spatially averaged version of ISMIP6 melt param.:
                meltfluxes = (param_dict['gamma0']['vec'][ii] * c_melt * TFs *
                              np.absolute(TFs) *
                              iceshelf_area) * rhoi / 1.0e12  # Gt/yr
                # interpolate deltaT value.  Use nan values outside of range
                # so out of range results get detected
                deltaT_vec[ii] = np.interp(param_dict['meltflux']['vec'][ii],
                                           meltfluxes, TFs,
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
            self.add_step(EnsembleMember(
                test_case=self, run_num=run_num,
                basal_fric_exp=param_dict['fric_exp']['vec'][run_num],
                mu_scale=param_dict['mu_scale']['vec'][run_num],
                stiff_scale=param_dict['stiff_scale']['vec'][run_num],
                von_mises_threshold=param_dict['von_mises_threshold']['vec'][run_num],  # noqa
                calv_spd_lim=param_dict['calv_limit']['vec'][run_num],
                gamma0=param_dict['gamma0']['vec'][run_num],
                meltflux=param_dict['meltflux']['vec'][run_num],
                deltaT=deltaT_vec[run_num]))
            # Note: do not add to steps_to_run, because ensemble_manager
            # will handle submitting and running the runs

        # Have 'compass run' only run the run_manager but not any actual runs.
        # This is because the individual runs will be submitted as jobs
        # by the ensemble manager.
        self.steps_to_run = ['ensemble_manager',]

    # no run() method is needed

    # no validate() method is needed
