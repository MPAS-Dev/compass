import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from compass.landice.tests.mesh_convergence.conv_analysis import ConvAnalysis


class Analysis(ConvAnalysis):
    """
    A step for visualizing the output from the advection convergence test case
    """
    def __init__(self, test_case, resolutions):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolutions : list of int
            The resolutions of the meshes that have been run
        """
        super().__init__(test_case=test_case, resolutions=resolutions)
        self.resolutions = resolutions
        self.add_output_file('convergence.png')

    def run(self):
        """
        Run this step of the test case
        """
        plt.switch_backend('Agg')
        resolutions = self.resolutions
        ncells_list = list()
        errors = list()
        for res in resolutions:
            rms_error, ncells = self.rmse(res)
            ncells_list.append(ncells)
            errors.append(rms_error)

        ncells = np.array(ncells_list)
        errors = np.array(errors)

        p = np.polyfit(np.log10(ncells), np.log10(errors), 1)
        conv = abs(p[0]) * 2.0

        error_fit = ncells**p[0] * 10**p[1]

        plt.loglog(ncells, error_fit, 'k')
        plt.loglog(ncells, errors, 'or')
        plt.annotate('Order of Convergence = {}'.format(np.round(conv, 3)),
                     xycoords='axes fraction', xy=(0.3, 0.95), fontsize=14)
        plt.xlabel('Number of Grid Cells', fontsize=14)
        plt.ylabel('L2 Norm', fontsize=14)
        section = self.config['mesh_convergence']
        duration = section.getfloat('duration')
        plt.title(f'Halfar convergence test, {duration} yrs')
        plt.savefig('convergence.png', bbox_inches='tight', pad_inches=0.1)

        section = self.config['halfar']
        conv_thresh = section.getfloat('conv_thresh')
        conv_max = section.getfloat('conv_max')

        if conv < conv_thresh:
            raise ValueError(f'order of convergence '
                             f' {conv} < min tolerence {conv_thresh}')

        if conv > conv_max:
            warnings.warn(f'order of convergence '
                          f'{conv} > max tolerence {conv_max}')

    def rmse(self, resolution):
        """
        Compute the RMSE for a given resolution

        Parameters
        ----------
        resolution : int
            The resolution of the (uniform) mesh in km

        Returns
        -------
        rms_error : float
            The root-mean-squared error

        ncells : int
            The number of cells in the mesh
        """
        res_tag = '{}km'.format(resolution)

        timelev = -1  # todo: determine a specified time level and err check

        ds = xr.open_dataset('{}_output.nc'.format(res_tag), decode_cf=False)
        # Find out what the ice density and flowA values for this run were.
        print('Collecting parameter values from the output file.')
        flowA = float(ds.config_default_flowParamA)
        print(f'Using a flowParamA value of: {flowA}')
        flow_n = float(ds.config_flowLawExponent)
        print(f'Using a flowLawExponent value of: {flow_n}')
        if flow_n != 3:
            sys.exit('Error: The Halfar script currently only supports a '
                     'flow law exponent of 3.')
        rhoi = ds.config_ice_density
        print(f'Using an ice density value of: {rhoi}')
        dynamicThickness = float(ds.config_dynamic_thickness)
        print(f'Dynamic thickness for this run = {dynamicThickness}')
        daysSinceStart = ds.daysSinceStart
        print(f'Using model time of {daysSinceStart/365.0} years')
        if ds.config_calendar_type != "noleap":
            sys.exit('Error: The Halfar script currently assumes a '
                     'gregorian_noleap calendar.')

        ncells = ds.sizes['nCells']
        thk = ds['thickness'].isel(Time=timelev)
        xCell = ds['xCell'].values
        yCell = ds['yCell'].values
        areaCell = ds['areaCell'].values

        dt = (daysSinceStart[timelev] - daysSinceStart[0]) * 3600.0 * 24.0

        thkHalfar = _halfar(dt, xCell, yCell, flowA, flow_n, rhoi)
        mask = np.where(np.logical_or(thk > 0.0, thkHalfar > 0.0))
        diff = thk - thkHalfar
        rms_error = ((diff[mask]**2 * areaCell[mask] /
                      areaCell[mask].sum()).sum())**0.5

        return rms_error, ncells


def _halfar(t, x, y, A, n, rho):
    # A   # s^{-1} Pa^{-3}
    # n   # Glen flow law exponent
    # rho # ice density kg m^{-3}

    # These constants should come from setup_dome_initial_conditions.py.
    # For now they are being hardcoded.
    R0 = 60000.0 * np.sqrt(0.125)   # initial dome radius
    H0 = 2000.0 * np.sqrt(0.125)    # initial dome thickness at center
    g = 9.80616  # gravity m/s/s -- value used by mpas_constants
    alpha = 1.0 / 9.0
    beta = 1.0 / 18.0
    Gamma = 2.0 / (n + 2.0) * A * (rho * g)**n

    # The IC file puts the center of the dome and the domain at (0,0)
    x0 = 0.0
    y0 = 0.0

    # NOTE: These constants assume n=3
    # they need to be generalized to allow other n's
    t0 = (beta / Gamma) * (7.0 / 4.0)**3 * (R0**4 / H0**7)
    t = t + t0
    t = t / t0

    H = np.zeros(len(x))
    for i in range(len(x)):
        r = np.sqrt((x[i] - x0)**2 + (y[i] - y0)**2)
        r = r / R0
        inside = max(0.0, 1.0 - (r / t**beta)**((n + 1.0) / n))

        H[i] = H0 * inside**(n / (2.0 * n + 1.0)) / t**alpha
    return H
