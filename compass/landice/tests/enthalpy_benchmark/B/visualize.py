import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.io import loadmat
from importlib.resources import path

from compass.io import add_input_file


def collect(testcase, step):
    """
    Update the dictionary of step properties

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case, which should not be
        modified here

    step : dict
        A dictionary of properties of this step, which can be updated
    """
    defaults = dict(cores=1, max_memory=1000, max_disk=1000, threads=1,
                    input_dir='run_model')
    for key, value in defaults.items():
        step.setdefault(key, value)

    step.setdefault('min_cores', step['cores'])

    add_input_file(step, filename='output.nc', target='../run_model/output.nc')

    filename = 'enthB_analy_result.mat'
    with path('compass.landice.tests.enthalpy_benchmark.B', filename) as \
            target:
        add_input_file(step, filename=filename, target=str(target))


# no setup function is needed


def run(step, test_suite, config, logger):
    """
    Run this step of the test case

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step from the ``collect()``
        function, with modifications from the ``setup()`` function.

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step
    """
    section = config['enthalpy_benchmark_viz']

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
    logger.info('Saved plot as {}'.format(plotname))

    if display_image:
        plt.show()
