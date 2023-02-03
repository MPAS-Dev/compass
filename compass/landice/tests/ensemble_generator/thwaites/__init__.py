from compass.validate import compare_variables
from compass.testcase import TestCase
from compass.landice.tests.ensemble_generator.ensemble_member \
        import EnsembleMember
from compass.landice.tests.ensemble_generator.ensemble_manager \
        import EnsembleManager
from importlib import resources
import numpy as np
import sys


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
        test_group : compass.landice.tests.thwaites.Thwaites
            The test group that this test case belongs to

        """
        name = 'thwaites_ensemble'
        super().__init__(test_group=test_group, name=name)

        # We don't want to initialize all the individual runs
        # So during init, we only add the run manager
        self.add_step(EnsembleManager(test_case=self))


    def configure(self):

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
        param_sample_number = param_array[:,0]
        param_unit_values = param_array[:,1:]
        max_samples = param_unit_values.shape[0]
        max_params = param_unit_values.shape[1]

        # Define parameters being sampled and their ranges
        # These options could eventually become cfg options
        # if that flexibility is desired.

        # basal fric exp
        basal_fric_param_idx = 0
        basal_fric_exp_range = [0.1, 0.333333]
        basal_fric_exp_vec = param_unit_values[:,basal_fric_param_idx] * \
                (basal_fric_exp_range[1] - basal_fric_exp_range[0]) + \
                basal_fric_exp_range[0]

        # von mises threshold stress
        von_mises_threshold_idx = 1
        von_mises_threshold_range = [150.0e3, 350.0e3]
        von_mises_threshold_vec = param_unit_values[:,von_mises_threshold_idx] * \
                (von_mises_threshold_range[1] - von_mises_threshold_range[0]) + \
                von_mises_threshold_range[0]

        # calving speed limit
        # Currently set to a constant value, but likely to be added later
        sec_in_yr = 3600.0 * 24.0 * 365.0
        calv_spd_lim_vec = 30.0e3 / sec_in_yr * np.ones((max_samples,))

        # add runs as steps based on the run range requested
        if self.end_run > max_samples:
            sys.exit("Error: end_run specified in config exceeds maximum sample "
                     "size available in param_vector_filename")
        for run_num in range(self.start_run, self.end_run+1):
            self.add_step(EnsembleMember(test_case=self, run_num=run_num,
                                         test_resources_location='compass.landice.tests.ensemble_generator.thwaites',
                                         basal_fric_exp=basal_fric_exp_vec[run_num],
                                         von_mises_threshold=von_mises_threshold_vec[run_num],
                                         calv_spd_lim=calv_spd_lim_vec[run_num]))
            # Note: do not add to steps_to_run, because ensemble_manager
            # will handle submitting and running the runs

        # Have compass run only run the run_manager but not any actual runs.
        # This is because the individual runs will be submitted as jobs
        # by the ensemble manager.
        self.steps_to_run = ['ensemble_manager',]


    # no run() method is needed

    # no validate() method is needed
