import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.io import loadmat
from importlib.resources import path

from compass.step import Step


class Visualize(Step):
    """
    A step for visualizing the output from a dome test case
    """
    def __init__(self, test_case):
        """
        Update the dictionary of step properties

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='visualize')

        for phase in range(1, 4):
            self.add_input_file(filename='output{}.nc'.format(phase),
                                target='../phase{}/output.nc'.format(phase))

        filename = 'enthA_analy_result.mat'
        with path('compass.landice.tests.enthalpy_benchmark.A', filename) as \
                target:
            self.add_input_file(filename=filename, target=str(target))

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        section = self.config['enthalpy_benchmark_viz']

        display_image = section.getboolean('display_image')

        if not display_image:
            plt.switch_backend('Agg')

        anaData = loadmat('enthA_analy_result.mat')
        basalMelt = anaData['basalMelt']

        SPY = 31556926

        years = list()
        basalMeanTs = list()
        basalMeanBmbs = list()
        basalMeanWaterThicknesses = list()

        for phase in range(1, 4):
            filename = 'output{}.nc'.format(phase)
            year, basalMeanT, basalMeanBmb, basalMeanWaterThickness = \
                _get_data(filename, SPY)
            years.append(year)
            basalMeanTs.append(basalMeanT)
            basalMeanBmbs.append(basalMeanBmb)
            basalMeanWaterThicknesses.append(basalMeanWaterThickness)

        year = np.concatenate(years)[1::] / 1000.0
        basalMeanT = np.concatenate(basalMeanTs)[1::]
        basalMeanBmb = np.concatenate(basalMeanBmbs)[1::]
        basalMeanWaterThickness = np.concatenate(basalMeanWaterThicknesses)[1::]

        plt.figure(1)
        plt.subplot(311)
        plt.plot(year, basalMeanT - 273.15)
        plt.ylabel(r'$T_{\rm b}$ ($^\circ \rm C$)')
        plt.text(10, -28, '(a)', fontsize=20)
        plt.grid(True)

        plt.subplot(312)
        plt.plot(year, -basalMeanBmb * SPY)
        plt.plot(basalMelt[1, :] / 1000.0, basalMelt[0, :], linewidth=2)
        plt.ylabel(r'$a_{\rm b}$ (mm a$^{-1}$ w.e.)')
        plt.text(10, -1.6, '(b)', fontsize=20)
        plt.grid(True)

        plt.subplot(313)
        plt.plot(year, basalMeanWaterThickness * 910.0 / 1000.0)
        plt.ylabel(r'$H_{\rm w}$ (m)')
        plt.xlabel('Year (ka)')
        plt.text(10, 8, '(c)', fontsize=20)
        plt.grid(True)

        # Create image plot
        plotname = 'enthalpy_A_results.png'
        plt.savefig(plotname, dpi=150)
        logger.info('Saved plot as {}'.format(plotname))

        if display_image:
            plt.show()


def _get_data(filename, SPY):
    G = 0.042
    kc = 2.1
    rhow = 1000.0
    Lw = 3.34e5

    dz = 2.5
    with Dataset(filename, 'r') as data:
        yr = data.variables['daysSinceStart'][:] / 365.0

        basalT = data.variables['basalTemperature'][:, :]
        basalMeanT = np.mean(basalT, axis=1)

        basalBmb = data.variables['groundedBasalMassBal'][:, :]
        basalMeanBmb = np.mean(basalBmb, axis=1)

        basalWaterThickness = data.variables['basalWaterThickness'][:, :]
        basalMeanWaterThickness = np.mean(basalWaterThickness, axis=1)

        T = data.variables['temperature'][:, :, :]
        TMean = np.mean(T, axis=1)
        TMean_nvert = TMean[:, -1]

        basalbmb = SPY * (G + kc * (TMean_nvert - basalMeanT) / dz) / (rhow * Lw)

        Hw = np.copy(basalbmb)
        for i in range(len(basalbmb)):
            Hw[i] = sum(basalbmb[0:i]) * 10

    return yr, basalMeanT, basalMeanBmb, basalMeanWaterThickness
