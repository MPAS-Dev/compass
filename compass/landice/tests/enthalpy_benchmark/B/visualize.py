import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.io import loadmat

from compass.step import Step


class Visualize(Step):
    """
    A step for visualizing the output from a dome test case
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='visualize')

        self.add_input_file(filename='output.nc',
                            target='../run_model/output.nc')

        self.add_input_file(
            filename='enthB_analy_result.mat',
            package='compass.landice.tests.enthalpy_benchmark.B')

    # no setup function is needed

    def run(self):
        """
        Run this step of the test case
        """
        section = self.config['enthalpy_benchmark_viz']

        display_image = section.getboolean('display_image')

        if not display_image:
            plt.switch_backend('Agg')

        anaData = loadmat('enthB_analy_result.mat')
        anaZ = anaData['enthB_analy_z']
        anaE = anaData['enthB_analy_E']
        anaT = anaData['enthB_analy_T']
        anaW = anaData['enthB_analy_omega']

        cp_ice = 2009.0
        # rho_ice = 910.0

        data = Dataset('output.nc', 'r')

        T = data.variables['temperature'][-1, :, :]
        horiMeanT = np.mean(T, axis=0)
        Ts = data.variables['surfaceTemperature'][-1, :]
        meanTs = np.mean(Ts)
        Tall = np.append(meanTs, horiMeanT)

        E = data.variables['enthalpy'][-1, :, :]
        horiMeanE = np.mean(E, axis=0)

        W = data.variables['waterFrac'][-1, :, :]
        horiMeanW = np.mean(W, axis=0)

        nz = len(data.dimensions['nVertLevels'])
        z = 1.0 - (np.arange(nz) + 1.0) / nz

        fsize = 14
        plt.figure(1)
        plt.subplot(1, 3, 1)
        plt.plot((horiMeanE / 910.0 + cp_ice * 50) / 1.0e3, z, label='MALI')
        plt.plot(anaE / 1000, anaZ, label='analytical')
        plt.xlabel(r'$E$ (10$^3$ J kg$^{-1}$)', fontsize=fsize)
        plt.ylabel(r'$z/H$', fontsize=fsize)
        plt.xticks(np.arange(92, 109, step=4), fontsize=fsize)
        plt.yticks(fontsize=fsize)
        plt.text(93, 0.05, 'a', fontsize=fsize)
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(Tall - 273.15, np.append(1, z))
        plt.plot(anaT - 273.15, anaZ)
        plt.xlabel(r'$T$ ($^\circ$C)', fontsize=fsize)
        # plt.ylabel('$\zeta$', fontsize=20)
        plt.xticks(np.arange(-3.5, 0.51, step=1), fontsize=fsize)
        plt.yticks(fontsize=fsize)
        plt.text(-3.2, 0.05, 'b', fontsize=fsize)
        plt.grid(True)
        # plt.gca().invert_yaxis()

        plt.subplot(1, 3, 3)
        plt.plot(horiMeanW * 100, z)
        plt.plot(anaW * 100, anaZ)
        plt.xlabel(r'$\omega$ (%)', fontsize=fsize)
        # plt.ylabel('$\zeta$',fontsize=20)
        # plt.xlim(-0.5,3)
        plt.xticks(np.arange(-0.5, 2.51, step=1), fontsize=fsize)
        plt.yticks(fontsize=fsize)
        plt.text(-0.3, 0.05, 'c', fontsize=fsize)
        plt.grid(True)

        plotname = 'enthalpy_B_results.png'
        plt.savefig(plotname, dpi=150)
        self.logger.info('Saved plot as {}'.format(plotname))

        if display_image:
            plt.show()
