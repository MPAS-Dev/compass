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

    for phase in range(1, 4):
        add_input_file(step, filename='output{}.nc'.format(phase),
                       target='../phase{}/output.nc'.format(phase))

    filename = 'enthA_analy_result.mat'
    with path('compass.landice.tests.enthalpy_benchmark.A', filename) as \
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
