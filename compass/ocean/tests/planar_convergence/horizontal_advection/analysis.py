import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings

from compass.ocean.tests.planar_convergence.conv_analysis import ConvAnalysis


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
            rms_error, ncells = self.rmse(res, variable='tracer1')
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
        plt.savefig('convergence.png', bbox_inches='tight', pad_inches=0.1)

        section = self.config['horizontal_advection']
        conv_thresh = section.getfloat('conv_thresh')
        conv_max = section.getfloat('conv_max')

        if conv < conv_thresh:
            raise ValueError(f'order of convergence '
                             f' {conv} < min tolerence {conv_thresh}')

        if conv > conv_max:
            warnings.warn(f'order of convergence '
                          f'{conv} > max tolerence {conv_max}')

    def rmse(self, resolution, variable):
        """
        Compute the RMSE for a given resolution

        Parameters
        ----------
        resolution : int
            The resolution of the (uniform) mesh in km

        variable : str
            The name of a variable in the output file to analyze.

        Returns
        -------
        rms_error : float
            The root-mean-squared error

        ncells : int
            The number of cells in the mesh
        """
        res_tag = '{}km'.format(resolution)

        ds = xr.open_dataset('{}_output.nc'.format(res_tag))
        init = ds[variable].isel(Time=0)
        final = ds[variable].isel(Time=-1)
        diff = final - init
        rms_error = np.sqrt((diff**2).mean()).values
        ncells = ds.sizes['nCells']
        return rms_error, ncells
