import sys
from importlib import resources

import numpy as np

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

        # Determine start and end run numbers being requested
        self.start_run = self.config.getint('ensemble', 'start_run')
        self.end_run = self.config.getint('ensemble', 'end_run')

        # Read pre-defined parameter vectors from a text file
        # These should have unit ranges
        param_file_name = self.config.get('ensemble',
                                          'param_vector_filename')
        with resources.open_text(
                'compass.landice.tests.ensemble_generator',
                param_file_name) as params_file:
            param_array = np.loadtxt(params_file, delimiter=',',
                                     skiprows=1)
        param_unit_values = param_array[:, 1:]
        max_samples = param_unit_values.shape[0]

        # Define parameters being sampled and their ranges
        # These options could eventually become cfg options
        # if that flexibility is desired.

        # basal fric exp
        basal_fric_param_idx = 0
        basal_fric_exp_range = [0.1, 0.333333]
        basal_fric_exp_vec = param_unit_values[:, basal_fric_param_idx] * \
            (basal_fric_exp_range[1] - basal_fric_exp_range[0]) + \
            basal_fric_exp_range[0]

        # von mises threshold stress
        von_mises_threshold_idx = 1
        von_mises_threshold_range = [100.0e3, 300.0e3]
        von_mises_threshold_vec = \
            param_unit_values[:, von_mises_threshold_idx] * \
            (von_mises_threshold_range[1] - von_mises_threshold_range[0]) + \
            von_mises_threshold_range[0]

        # calving speed limit
        # Currently set to a constant value, but likely to be added later
        sec_in_yr = 3600.0 * 24.0 * 365.0
        calv_spd_lim_vec = 30.0e3 / sec_in_yr * np.ones((max_samples,))

        # gamma0
        gamma0_idx = 2
        # gamma0 range spans 5th pct MeanAnt through 95th pct PIGL
        # from Jourdain et al. (2020)
        gamma0_range = [9620.0, 471000.0]
        gamma0_vec = param_unit_values[:, gamma0_idx] * \
            (gamma0_range[1] - gamma0_range[0]) + \
            gamma0_range[0]

        # melt flux
        meltflux_idx = 3
        # melt flux range (Gt/yr) is Rignot et al. (2013) estimate with
        # uncertainty.  Rignot extent ice-shelf area is close to our initial
        # condition.  An area adjustment is added, assuming mean melt rate.
        meltflux_range = [97.5 - 7.0, 97.5 + 7.0]
        iceshelf_area_obs = 4411.0e6  # m2
        meltflux_vec = param_unit_values[:, meltflux_idx] * \
            (meltflux_range[1] - meltflux_range[0]) + \
            meltflux_range[0]

        # deltaT
        section = self.config['ensemble']
        input_file_path = section.get('input_file_path')
        TF_file_path = section.get('TF_file_path')
        mean_TF, iceshelf_area = calc_mean_TF(input_file_path, TF_file_path)
        # Adjust observed melt flux for ice-shelf area in our initial condition
        print(f'IS area: model={iceshelf_area}, rignot={iceshelf_area_obs}')
        area_correction = iceshelf_area / iceshelf_area_obs
        print(f"Ice-shelf area correction is {area_correction}.")
        if (np.absolute(area_correction - 1.0) > 0.2):
            sys.exit("ERROR: ice-shelf area correction is larger than 20%. "
                     "Check data consistency before proceeding.")
        meltflux_vec *= iceshelf_area / iceshelf_area_obs
        TFs = np.linspace(-5.0, 10.0, num=int(15.0 / 0.01))
        rhoi = 910.0
        rhosw = 1028.0
        cp_seawater = 3.974e3
        latent_heat_ice = 335.0e3
        c_melt = (rhosw * cp_seawater / (rhoi * latent_heat_ice))**2
        deltaT_vec = np.zeros(max_samples)
        for ii in range(self.start_run, self.end_run + 1):
            meltfluxes = (gamma0_vec[ii] * c_melt * TFs * np.absolute(TFs) *
                          iceshelf_area) * rhoi / 1.0e12  # Gt/yr
            deltaT_vec[ii] = np.interp(meltflux_vec[ii], meltfluxes, TFs,
                                       left=np.nan, right=np.nan) - mean_TF

        # add runs as steps based on the run range requested
        if self.end_run > max_samples:
            sys.exit("Error: end_run specified in config exceeds maximum "
                     "sample size available in param_vector_filename")
        for run_num in range(self.start_run, self.end_run + 1):
            self.add_step(EnsembleMember(test_case=self, run_num=run_num,
                          test_resources_location='compass.landice.tests.ensemble_generator.thwaites',  # noqa
                          basal_fric_exp=basal_fric_exp_vec[run_num],
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
