"""
Select the melt parameter from the MALI ensemble, per the ISMIP7 protocol.
"""

import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf

from compass.landice.tests.ismip7_calibration import datasets
from compass.landice.tests.ismip7_calibration.ais.aggregate import (
    unit_aggregates,
)
from compass.landice.tests.ismip7_calibration.configure import (
    objective_options,
    parameter_name,
    parameter_values,
    weighting,
)
from compass.landice.tests.ismip7_calibration.objective import (
    build_toolbox_terms,
    run_optimisation,
)
from compass.step import Step


class Calibrate(Step):
    """
    A step that runs the ISMIP7 parameter selection on the MALI ensemble.

    The protocol draws random term weights and random targets within their
    uncertainties, and for each draw picks the parameter that minimises
    ``I = sum_i a_i J_i / median(J_i)``.  The 5th, 50th and 95th percentiles
    of the resulting distribution are what the ISMIP7 projections need.

    Because the objective **normalises each term by its own median** over the
    parameter ensemble, a minimised ``I`` is not comparable between melt
    forms -- only between parameter values within one form.  This step
    therefore reports a distribution per melt form and never a cross-form
    comparison of the objective.

    Attributes
    ----------
    melt_forms : list of str
        The melt forms to calibrate
    """

    def __init__(self, test_case, melt_forms):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_calibration.ais.Ais
            The test case this step belongs to

        melt_forms : list of str
            The melt forms to calibrate
        """
        super().__init__(test_case=test_case, name='calibrate')
        self.melt_forms = melt_forms
        for melt_form in melt_forms:
            self.add_input_file(
                filename=f'aggregates_{melt_form}.nc',
                target=f'../aggregate/aggregates_{melt_form}.nc')
            self.add_output_file(filename=f'calibration_{melt_form}.nc')

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config

        base_path = config.get('ismip7_calibration', 'base_path_ismip7')
        targets = datasets.load_targets(base_path)
        t3_models, t4_regions, t4_years = weighting(config)
        sample_size, seed = objective_options(config)

        logger.info(f'J3 weighting: '
                    f'{"all models" if t3_models is None else t3_models}')
        logger.info(f'J4 weighting: '
                    f'{"PIG and Dotson" if t4_regions is None else t4_regions}'
                    f', '
                    f'{"all years" if t4_years is None else t4_years}')

        for melt_form in self.melt_forms:
            name = parameter_name(melt_form)
            values = parameter_values(config, melt_form)
            units = unit_aggregates(f'aggregates_{melt_form}.nc')

            terms = build_toolbox_terms(units, targets, values,
                                        t3_models=t3_models,
                                        t4_regions=t4_regions,
                                        t4_years=t4_years)

            logger.info(f'Sampling the objective function {sample_size} '
                        f'times for the {melt_form} form...')
            result = run_optimisation(terms, values, sample_size=sample_size,
                                      seed=seed)

            _write(result, values, melt_form, name,
                   f'calibration_{melt_form}.nc')
            _report(result, melt_form, name, logger)


def _write(result, values, melt_form, name, filename):
    """Write one parameter distribution to a file."""
    ds = xr.Dataset()
    ds['min_p1'] = ('sample', result['min_p1'])
    ds['parameter_values'] = ('parameter', np.asarray(values))
    for key in ('p5', 'median', 'p95', 'mode'):
        ds[key] = float(result[key])
    ds.attrs['melt_form'] = melt_form
    ds.attrs['parameter'] = name
    ds.attrs['note'] = (
        'The objective normalises each term by its own median over the '
        'parameter ensemble, so the minimised objective is not comparable '
        'between melt forms -- only between parameter values within one '
        'form.')
    write_netcdf(ds, filename)


def _report(result, melt_form, name, logger):
    """Log the selected percentiles."""
    logger.info('')
    logger.info(f'{melt_form}: {name} percentiles on the MALI mesh')
    for key in ('p5', 'median', 'p95'):
        logger.info(f'  {key:8s} {result[key]:12.5e}')
    logger.info(f'  {"mode":8s} {result["mode"]:12.5e}')
    logger.info('')
