"""
Report the calibrated parameter distributions and per-term diagnostics.
"""

import matplotlib
import numpy as np
import xarray as xr

from compass.landice.tests.ismip7_calibration.configure import parameter_name
from compass.step import Step

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


class Report(Step):
    """
    A step that plots and tabulates the calibration result.

    Attributes
    ----------
    melt_forms : list of str
        The melt forms that were calibrated
    """

    def __init__(self, test_case, melt_forms):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_calibration.ais.Ais
            The test case this step belongs to

        melt_forms : list of str
            The melt forms that were calibrated
        """
        super().__init__(test_case=test_case, name='report')
        self.melt_forms = melt_forms

        for melt_form in melt_forms:
            self.add_input_file(
                filename=f'calibration_{melt_form}.nc',
                target=f'../calibrate/calibration_{melt_form}.nc')
            self.add_output_file(
                filename=f'parameter_distribution_{melt_form}.png')
        self.add_input_file(filename='shelf_area.nc',
                            target='../aggregate/shelf_area.nc')
        self.add_output_file(filename='calibration_summary.txt')

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        lines = []

        lines.append('ISMIP7 melt-module calibration on the MALI mesh')
        lines.append('=' * 62)
        lines.append('')

        for melt_form in self.melt_forms:
            name = parameter_name(melt_form)
            with xr.open_dataset(f'calibration_{melt_form}.nc') as ds:
                percentiles = {key: float(ds[key])
                               for key in ('p5', 'median', 'p95', 'mode')}
                samples = ds['min_p1'].values
                values = ds['parameter_values'].values

            lines.append(f'{melt_form}: {name}')
            for key in ('p5', 'median', 'p95', 'mode'):
                lines.append(f'  {key:8s} {percentiles[key]:12.5e}')
            lines.append('')

            _plot_distribution(samples, values, percentiles, melt_form, name)

        lines.append('The objective normalises each term by its own median '
                     'over the parameter')
        lines.append('ensemble, so the minimised objective is NOT comparable '
                     'between melt forms.')
        lines.append('Only parameter values within one form can be compared '
                     'by it.')
        lines.append('')

        lines.extend(_shelf_area_lines())

        text = '\n'.join(lines) + '\n'
        with open('calibration_summary.txt', 'w') as handle:
            handle.write(text)
        for line in lines:
            logger.info(line)


def _plot_distribution(samples, values, percentiles, melt_form, name):
    """Plot the distribution of optimal parameter values."""
    step = np.diff(values).min()
    edges = np.append(values - 0.5 * step, values[-1] + 0.5 * step)

    fig, axis = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    axis.hist(samples, bins=edges, color='0.7', edgecolor='none')
    for key, color, style in (('p5', 'C0', '--'), ('median', 'C3', '-'),
                              ('p95', 'C0', '--')):
        axis.axvline(percentiles[key], color=color, linestyle=style,
                     label=f'{key} = {percentiles[key]:.4g}')
    axis.set_xlabel(f'{name} ({melt_form})')
    axis.set_ylabel('draws')
    axis.set_title(f'Optimal {name} over 100,000 draws of the term weights\n'
                   f'and targets, {melt_form} form on the MALI mesh')
    axis.legend()
    fig.savefig(f'parameter_distribution_{melt_form}.png', dpi=150)
    plt.close(fig)


def _shelf_area_lines():
    """Tabulate modelled against observed ice-shelf area."""
    lines = ['Ice-shelf area by ISMIP7 basin, 10^3 km^2',
             '-' * 62,
             f'  {"basin":>5s} {"MALI":>10s} {"observed":>10s} '
             f'{"ratio":>8s}']
    with xr.open_dataset('shelf_area.nc') as ds:
        modelled = ds['modelled_shelf_area']
        observed = ds['observed_shelf_area']
        for basin in modelled.basins.values:
            mali = float(modelled.sel(basins=basin))
            obs = float(observed.sel(basins=basin))
            ratio = mali / obs if obs > 0 else float('nan')
            lines.append(f'  {int(basin):5d} {mali:10.1f} {obs:10.1f} '
                         f'{ratio:8.2f}')
        total_mali = float(modelled.sum())
        total_obs = float(observed.sum())
        lines.append(f'  {"total":>5s} {total_mali:10.1f} {total_obs:10.1f} '
                     f'{total_mali / total_obs:8.2f}')
    lines.append('')
    lines.append('J1, J2 and J4 are integrated melt, so a shelf-area '
                 'mismatch enters the')
    lines.append('calibrated parameter directly.  J3 is a basin mean and is '
                 'much less sensitive.')
    lines.append('')
    return lines
