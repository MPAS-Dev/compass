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
from compass.landice.tests.ensemble_generator.ensemble_template import (
    get_spinup_template_package,
)
from compass.testcase import TestCase


class SpinupEnsemble(TestCase):
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
        name = 'spinup_ensemble'
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

        config = self.config
        resource_module = get_spinup_template_package(config)

        section = config['ensemble_generator']
        spinup_section_name = 'spinup_ensemble'
        if not config.has_section(spinup_section_name):
            raise ValueError(
                f"Missing required config section '{spinup_section_name}'.")
        spinup_section = config[spinup_section_name]
        parameter_section_name = 'ensemble.parameters'
        if not config.has_section(parameter_section_name):
            raise ValueError(
                f"Missing required config section '{parameter_section_name}'.")
        param_section = config[parameter_section_name]

        # Determine start and end run numbers being requested
        self.start_run = section.getint('start_run')
        self.end_run = section.getint('end_run')

        parameter_specs = _get_parameter_specs(param_section)

        # Determine how many parameters are being sampled.
        n_params = len(parameter_specs)
        if n_params == 0:
            sys.exit("ERROR: At least one parameter must be specified.")

        max_samples = section.getint('max_samples')
        if max_samples < self.end_run:
            sys.exit("ERROR: max_samples is exceeded by end_run")
        sampling_method = section.get('sampling_method')
        parameter_specs = _populate_parameter_vectors(
            parameter_specs=parameter_specs,
            sampling_method=sampling_method,
            max_samples=max_samples)
        if config.has_option('spinup_ensemble', 'fric_samples_file'):
            print("Including friction samples from file. One friction "
                  "sample will be mapped to each run in the ensemble.")

        spec_by_name = {spec['name']: spec for spec in parameter_specs}

        deltaT_vec = _compute_delta_t_vec(
            config=config, spinup_section=spinup_section,
            spec_by_name=spec_by_name,
            max_samples=max_samples, start_run=self.start_run,
            end_run=self.end_run)

        _add_member_steps(
            test_case=self, parameter_specs=parameter_specs,
            spec_by_name=spec_by_name, deltaT_vec=deltaT_vec,
            resource_module=resource_module, max_samples=max_samples)

        # Have 'compass run' only run the run_manager but not any actual runs.
        # This is because the individual runs will be submitted as jobs
        # by the ensemble manager.
        self.steps_to_run = ['ensemble_manager',]

    # no run() method is needed

    # no validate() method is needed


def _get_parameter_specs(section):
    """Build parameter specification dictionaries from config options.

    Parameters with an ``nl.`` prefix are treated as namelist parameters and
    include one or more target namelist option names. Other parameters are
    interpreted as supported special parameters (for example ``gamma0``).

    Returns
    -------
    list of dict
        Ordered parameter metadata with sampled bounds and placeholders for
        populated sample vectors.
    """
    specs = []
    special_params = {'fric_exp', 'mu_scale', 'stiff_scale',
                      'gamma0', 'meltflux'}

    for option_name, raw_value in section.items():
        if option_name.endswith('.option_name'):
            continue
        parameter_name = option_name
        bounds = _parse_range(raw_value, parameter_name)

        if parameter_name.startswith('nl.'):
            option_key = f'{parameter_name}.option_name'
            if option_key not in section:
                raise ValueError(
                    f"Namelist parameter '{parameter_name}' must define "
                    f"'{option_key}'.")
            namelist_options = _split_entries(section[option_key])
            if len(namelist_options) == 0:
                raise ValueError(
                    f"Namelist parameter '{parameter_name}' has no "
                    "option names configured.")
            specs.append({
                'name': parameter_name,
                'type': 'namelist',
                'run_info_name': parameter_name[len('nl.'):],
                'option_names': namelist_options,
                'min': bounds[0],
                'max': bounds[1],
                'vec': None
            })
        else:
            if parameter_name not in special_params:
                raise ValueError(
                    f"Unsupported special parameter '{parameter_name}'.")
            specs.append({
                'name': parameter_name,
                'type': 'special',
                'min': bounds[0],
                'max': bounds[1],
                'vec': None
            })

    return specs


def _populate_parameter_vectors(parameter_specs, sampling_method,
                                max_samples):
    """Generate and scale samples to each parameter range.

    This function updates each ``spec['vec']`` in ``parameter_specs`` and
    returns the same list for explicit readability at call site.
    ``sobol`` creates a space-filling sequence in unit space,
    ``uniform`` creates linearly spaced samples, and ``log-uniform`` samples
    linearly in log10 space (requiring strictly positive bounds).

    Returns
    -------
    list of dict
        The same ``parameter_specs`` list with each ``spec['vec']`` populated.
    """
    n_params = len(parameter_specs)
    if sampling_method == 'sobol':
        print(f"Generating Sobol sequence for {n_params} parameter(s)")
        sampler = qmc.Sobol(d=n_params, scramble=True, seed=4)
        param_unit_values = sampler.random(n=max_samples)
    elif sampling_method in {'uniform', 'log-uniform'}:
        print(f"Generating {sampling_method} sampling for "
              f"{n_params} parameter(s)")
        samples = np.linspace(0.0, 1.0, max_samples).reshape(-1, 1)
        param_unit_values = np.tile(samples, (1, n_params))
    else:
        sys.exit("ERROR: Unsupported sampling method specified.")

    if sampling_method == 'log-uniform':
        for spec in parameter_specs:
            if spec['min'] <= 0.0 or spec['max'] <= 0.0:
                sys.exit(
                    "ERROR: log-uniform sampling requires positive min/max "
                    f"for parameter '{spec['name']}'.")

    for idx, spec in enumerate(parameter_specs):
        print('Including parameter ' + spec['name'])
        if sampling_method == 'log-uniform':
            log_min = np.log10(spec['min'])
            log_max = np.log10(spec['max'])
            spec['vec'] = 10.0 ** (param_unit_values[:, idx] *
                                   (log_max - log_min) + log_min)
        else:
            spec['vec'] = param_unit_values[:, idx] * \
                (spec['max'] - spec['min']) + spec['min']
    return parameter_specs


def _compute_delta_t_vec(config, spinup_section, spec_by_name, max_samples,
                         start_run, end_run):
    """Compute per-run ``deltaT`` values when ``meltflux`` is active.

    If ``meltflux`` is not sampled, this returns a list of ``None`` values.
    When active, the function applies ice-shelf area correction to sampled
    melt flux and interpolates the ``deltaT`` needed to match each target
    melt flux over the requested run range.

    Returns
    -------
    list or numpy.ndarray
        ``[None] * max_samples`` when ``meltflux`` is inactive, otherwise a
        ``numpy.ndarray`` containing per-run ``deltaT`` values.
    """
    if 'meltflux' not in spec_by_name:
        return [None] * max_samples

    if 'gamma0' not in spec_by_name:
        sys.exit("ERROR: parameter 'meltflux' requires 'gamma0'.")
    if not config.has_option('spinup_ensemble', 'iceshelf_area_obs'):
        sys.exit(
            "ERROR: parameter 'meltflux' requires "
            "'iceshelf_area_obs' in [spinup_ensemble].")

    iceshelf_area_obs = spinup_section.getfloat('iceshelf_area_obs')
    input_file_path = spinup_section.get('input_file_path')
    TF_file_path = spinup_section.get('TF_file_path')
    mean_TF, iceshelf_area = calc_mean_TF(input_file_path, TF_file_path)

    print(f'IS area: model={iceshelf_area}, Obs={iceshelf_area_obs}')
    area_correction = iceshelf_area / iceshelf_area_obs
    print(f"Ice-shelf area correction is {area_correction}.")
    if np.absolute(area_correction - 1.0) > 0.2:
        print("WARNING: ice-shelf area correction is larger than "
              "20%. Check data consistency before proceeding.")

    spec_by_name['meltflux']['vec'] *= area_correction

    rhoi = 910.0
    rhosw = 1028.0
    cp_seawater = 3.974e3
    latent_heat_ice = 335.0e3
    c_melt = (rhosw * cp_seawater / (rhoi * latent_heat_ice))**2
    TFs = np.linspace(-5.0, 10.0, num=int(15.0 / 0.01))
    deltaT_vec = np.zeros(max_samples)
    for ii in range(start_run, end_run + 1):
        meltfluxes = (spec_by_name['gamma0']['vec'][ii] * c_melt *
                      TFs * np.absolute(TFs) * iceshelf_area) * \
            rhoi / 1.0e12  # Gt/yr
        deltaT_vec[ii] = np.interp(
            spec_by_name['meltflux']['vec'][ii], meltfluxes, TFs,
            left=np.nan, right=np.nan) - mean_TF
        if np.isnan(deltaT_vec[ii]):
            sys.exit("ERROR: interpolated deltaT out of range. "
                     "Adjust definition of 'TFs'")

    return deltaT_vec


def _build_namelist_values(parameter_specs, run_num):
    """For parameter specs of type 'namelist',
    collect namelist option values for a given run number
    and save them in a dictionary keyed by namelist option name.
    These will be applied when the runs are set up.

    Returns
    -------
    tuple of dict
        ``(namelist_option_values, namelist_parameter_values)`` for the
        requested ``run_num``.
    """
    namelist_option_values = {}
    namelist_parameter_values = {}

    for spec in parameter_specs:
        if spec['type'] != 'namelist':
            continue
        value = spec['vec'][run_num]
        for namelist_option in spec['option_names']:
            namelist_option_values[namelist_option] = value
        namelist_parameter_values[spec['run_info_name']] = value

    return namelist_option_values, namelist_parameter_values


def _add_member_steps(test_case, parameter_specs, spec_by_name, deltaT_vec,
                      resource_module, max_samples):
    """Create and register ``EnsembleMember`` steps for requested runs.

    This helper assembles namelist and special-parameter values for each run
    and adds one member step per run to ``test_case``.
    """
    if test_case.end_run > max_samples:
        sys.exit("Error: end_run specified in config exceeds maximum "
                 "sample size available in param_vector_filename")

    print("--- Identifying required parameters is complete ---")
    for run_num in range(test_case.start_run, test_case.end_run + 1):
        namelist_option_values, namelist_parameter_values = \
            _build_namelist_values(parameter_specs, run_num)

        fric_exp = _get_special_value(spec_by_name, 'fric_exp', run_num)
        mu_scale = _get_special_value(spec_by_name, 'mu_scale', run_num)
        stiff_scale = _get_special_value(spec_by_name, 'stiff_scale',
                                         run_num)
        gamma0 = _get_special_value(spec_by_name, 'gamma0', run_num)
        meltflux = _get_special_value(spec_by_name, 'meltflux', run_num)

        test_case.add_step(EnsembleMember(
            test_case=test_case, run_num=run_num,
            basal_fric_exp=fric_exp,
            mu_scale=mu_scale,
            stiff_scale=stiff_scale,
            gamma0=gamma0,
            meltflux=meltflux,
            deltaT=deltaT_vec[run_num],
            namelist_option_values=namelist_option_values,
            namelist_parameter_values=namelist_parameter_values,
            resource_module=resource_module))
        # Note: do not add to steps_to_run, because ensemble_manager
        # will handle submitting and running the runs


def _split_entries(raw):
    """Split comma- or whitespace-delimited config lists.

    Backslash-newline sequences used for line continuation are stripped so
    that multi-line values are treated as a single logical line. Remaining
    backslashes are also removed to avoid spurious option tokens.

    Returns
    -------
    list of str
        Non-empty parsed entries.
    """
    cleaned = raw.replace('\\\r\n', ' ').replace('\\\n', ' ')
    cleaned = cleaned.replace('\\', ' ')
    return [entry for entry in cleaned.replace(',', ' ').split() if entry]


def _parse_range(raw, parameter_name):
    """Parse parameter min,max bounds from a comma-delimited value.

    Returns
    -------
    tuple of float
        ``(min_value, max_value)`` parsed from ``raw``.
    """
    values = [entry.strip() for entry in raw.split(',') if entry.strip()]
    if len(values) != 2:
        raise ValueError(
            f"Parameter '{parameter_name}' must contain exactly "
            "two comma-separated values.")
    return float(values[0]), float(values[1])


def _get_special_value(spec_by_name, name, run_num):
    """Get sampled value for a special parameter or ``None`` if inactive.

    Returns
    -------
    float or None
        Sampled value for ``name`` at ``run_num`` when present.
    """
    if name not in spec_by_name:
        return None
    return spec_by_name[name]['vec'][run_num]
