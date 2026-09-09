from compass.landice.tests.ismip7_calibration import datasets
from compass.landice.tests.ismip7_calibration.ais.aggregate import Aggregate
from compass.landice.tests.ismip7_calibration.ais.calibrate import Calibrate
from compass.landice.tests.ismip7_calibration.ais.fit_delta_t import FitDeltaT
from compass.landice.tests.ismip7_calibration.ais.remap_forcing import (
    RemapForcing,
)
from compass.landice.tests.ismip7_calibration.ais.remap_masks import RemapMasks
from compass.landice.tests.ismip7_calibration.ais.report import Report
from compass.landice.tests.ismip7_calibration.ais.run_state import RunState
from compass.landice.tests.ismip7_calibration.ais.verify_melt import VerifyMelt
from compass.landice.tests.ismip7_calibration.configure import (
    check_options,
    melt_forms,
)
from compass.testcase import TestCase
from compass.validate import compare_variables

#: the ocean state the verification and the dT_b fit use
REFERENCE_STATE = 'climatology'

#: multiples of the reference melt parameter used to measure the linearity
#: that the one-run-per-ocean-state ensemble relies on.  Only the 'ismip7'
#: form is scaled: its parameter is a namelist option, while the ISMIP6
#: gamma0 is read from an input file.
LINEARITY_SCALES = (0.5, 2.0)


class Ais(TestCase):
    """
    A test case that calibrates MALI's sub-shelf melt parameterization on an
    Antarctic MALI mesh, following the ISMIP7 protocol.

    The steps, in dependency order:

    ``remap_masks``
        ISMIP7 basins, buttressing bins, the floating mask and the
        PIG/Dotson regions, onto the MALI mesh.

    ``remap_forcing``
        The calibration thermal forcing for each ocean state, onto the MALI
        mesh.

    ``<melt_form>_<state>``
        One single-timestep MALI melt diagnostic per ocean state and melt
        form.  **Not** one per parameter value: melt is exactly proportional
        to the melt parameter, so the parameter sweep is a scaling of one
        run.  That is what makes this 28 runs per form rather than about
        1300.

    ``verify_melt``
        MALI's melt against an independent Python implementation, its
        vertical interpolation against an independent one, and the linearity
        that the previous point relies on.

    ``aggregate``
        Melt to basins, buttressing bins and shelf regions, area-weighted.

    ``calibrate``
        The 100,000-sample parameter selection, giving the percentiles the
        ISMIP7 projections need.

    ``fit_delta_t``
        The per-basin correction dT_b, fitted **after** parameter selection
        per protocol Sect. 4.2.1 option 2.

    ``report``
        Plots and a summary table.

    Attributes
    ----------
    melt_forms : list of str
        The melt forms being calibrated

    states : list of compass.landice.tests.ismip7_calibration.datasets.OceanState
        The ocean states in the ensemble
    """  # noqa: E501

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ismip7_calibration.Ismip7Calibration
            The test group that this test case belongs to
        """  # noqa: E501
        name = 'ais'
        super().__init__(test_group=test_group, name=name, subdir=name)
        self.melt_forms = []
        self.states = []

    def configure(self):
        """
        Add a step per ocean state and melt form, once the config is known
        """
        config = self.config
        check_options(config, ['base_path_ismip7', 'base_path_mali',
                               'mali_mesh_file', 'mali_mesh_name',
                               'graph_file_prefix'])

        section = config['ismip7_calibration']
        base_path = section.get('base_path_ismip7')
        subset = section.get('ocean_state_subset')

        self.melt_forms = melt_forms(config)
        self.states = datasets.ocean_states(base_path, subset=subset)

        self.add_step(RemapMasks(test_case=self))
        self.add_step(RemapForcing(test_case=self))

        for melt_form in self.melt_forms:
            for state in self.states:
                self.add_step(RunState(
                    test_case=self, state_name=state.name,
                    melt_form=melt_form,
                    subdir=f'{melt_form}_{state.name}'))

        # extra runs of one ocean state at other melt parameters, so that the
        # linearity the ensemble design relies on is measured in MALI rather
        # than argued from the code
        verify_form = self.melt_forms[0]
        scales = LINEARITY_SCALES if verify_form == 'ismip7' else ()
        for scale in scales:
            self.add_step(RunState(
                test_case=self, state_name=REFERENCE_STATE,
                melt_form=verify_form,
                subdir=f'{verify_form}_{REFERENCE_STATE}_x{scale:g}',
                parameter_scale=scale))

        self.add_step(VerifyMelt(test_case=self, melt_form=verify_form,
                                 state_name=REFERENCE_STATE,
                                 linearity_scales=scales))
        self.add_step(Aggregate(test_case=self, melt_forms=self.melt_forms,
                                states=self.states))
        self.add_step(Calibrate(test_case=self, melt_forms=self.melt_forms))
        self.add_step(FitDeltaT(test_case=self, melt_forms=self.melt_forms,
                                state_name=REFERENCE_STATE))
        self.add_step(Report(test_case=self, melt_forms=self.melt_forms))

    def validate(self):
        """
        Compare the calibration against a baseline, if one was provided
        """
        variables = ['p5', 'median', 'p95', 'mode']
        for melt_form in self.melt_forms:
            compare_variables(
                test_case=self, variables=variables,
                filename1=f'calibrate/calibration_{melt_form}.nc')
        compare_variables(
            test_case=self,
            variables=['modelled_shelf_area', 'observed_shelf_area'],
            filename1='aggregate/shelf_area.nc')
