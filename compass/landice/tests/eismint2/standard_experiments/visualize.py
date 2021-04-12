import datetime
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from compass.step import Step


class Visualize(Step):
    """
    A step for visualizing the output from a EISMINT2 test case
    """
    def __init__(self, test_case):
        """
        Update the dictionary of step properties

        Parameters
        ----------
        test_case : compass.landice.tests.eismint2.standard_experiments.StandardExperiments
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='visualize')

        # depending on settings, this may produce no outputs, so we won't add
        # any

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger
        experiment = config.get('eismint2_viz', 'experiment')

        if ',' in experiment:
            experiments = [exp.strip() for exp in experiment.split(',')]
        else:
            experiments = [experiment]

        for experiment in experiments:
            logger.info('Plotting Experiment {}'.format(experiment))
            visualize_eismint2(config, logger, experiment)


def visualize_eismint2(config, logger, experiment):
    """
    Plot the output from an EISMINT2 experiment

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step

    experiment : {'a', 'b', 'c', 'd', 'f', 'g'}
        The name of the experiment
    """

    section = config['eismint2_viz']
    save_images = section.getboolean('save_images')
    hide_figs = section.getboolean('hide_figs')

    filename = '../experiment_{}/output.nc'.format(experiment)

    # open supplied MPAS output file and get variables needed
    filein = netCDF4.Dataset(filename, 'r')
    xCell = filein.variables['xCell'][:]/1000.0
    yCell = filein.variables['yCell'][:]/1000.0
    xtime = filein.variables['xtime'][:]
    nCells = len(filein.dimensions['nCells'])
    nVertLevels = len(filein.dimensions['nVertLevels'])
    years = _xtime_get_year(xtime)

    thickness = filein.variables['thickness']
    basalTemperature = filein.variables['basalTemperature']
    basalPmpTemperature = filein.variables['basalPmpTemperature']
    flwa = filein.variables['flowParamA']
    uReconstructX = filein.variables['uReconstructX']
    uReconstructY = filein.variables['uReconstructY']
    areaCell = filein.variables['areaCell'][:]
    layerThicknessFractions = filein.variables['layerThicknessFractions'][:]

    # Use final time
    timelev = -1
    logger.info('Using final model time of {} \n'.format(
        xtime[timelev, :].tostring().strip().decode('utf-8')))

    # ================
    # ================
    # Plot the results
    # ================
    # ================

    # ================
    # BASAL TEMPERATURE MAP
    # ================

    # make an educated guess about how big the markers should be.
    if nCells**0.5 < 100.0:
        markersize = max(int(round(3600.0/(nCells**0.5))), 1)
        # use hexes if the points are big enough, otherwise just dots
        markershape = 'h'
    else:
        markersize = max(int(round(1800.0/(nCells**0.5))), 1)
        markershape = '.'
    logger.info('Using a markersize of {}'.format(markersize))

    fig = plt.figure(1, facecolor='w')
    fig.suptitle('Payne et al. Fig. 1, 3, 6, 9, or 11', fontsize=10, fontweight='bold')

    iceIndices = np.where(thickness[timelev, :] > 10.0)[0]
    plt.scatter(xCell[iceIndices], yCell[iceIndices], markersize,
                c=np.array([[0.8, 0.8, 0.8], ]), marker=markershape,
                edgecolors='none')

    # add contours of ice temperature over the top
    basalTemp = basalTemperature[timelev, :]
    # fill places below dynamic limit with non-ice value of 273.15
    basalTemp[np.where(thickness[timelev, :] < 10.0)] = 273.15
    _contour_mpas(basalTemp, nCells, xCell, yCell,
                  contour_levs=np.linspace(240.0, 275.0, 8))

    plt.axis('equal')
    plt.title('Modeled basal temperature (K) \n at time {}'.format(
        netCDF4.chartostring(xtime)[timelev].strip()))
    plt.xlim((0.0, 1500.0))
    plt.ylim((0.0, 1500.0))
    plt.xlabel('X position (km)')
    plt.ylabel('Y position (km)')

    if save_images:
        plt.savefig('EISMINT2-{}-basaltemp.png'.format(experiment), dpi=150)

    # ================
    # STEADY STATE MAPS -  panels b and c are switched and with incorrect units in the paper
    # ================
    fig = plt.figure(2, facecolor='w', figsize=(12, 6), dpi=72)
    fig.suptitle('Payne et al. Fig. 2 or 4', fontsize=10, fontweight='bold')

    # ================
    # panel a - thickness
    ax1 = fig.add_subplot(131)

    plt.scatter(xCell[iceIndices], yCell[iceIndices], markersize,
                c=np.array([[0.8, 0.8, 0.8], ]), marker=markershape,
                edgecolors='none')

    # add contours of ice thickness over the top
    contour_intervals = np.linspace(0.0, 5000.0,  int(5000.0/250.0)+1)
    _contour_mpas(thickness[timelev, :], nCells, xCell, yCell,
                  contour_levs=contour_intervals)

    plt.title('Final thickness (m)')
    ax1.set_aspect('equal')
    plt.xlabel('X position (km)')
    plt.ylabel('Y position (km)')

    # ================
    # panel c - flux
    ax = fig.add_subplot(133, sharex=ax1, sharey=ax1)

    flux = np.zeros((nCells,))
    for k in range(nVertLevels):
        speedLevel = (uReconstructX[timelev, :, k:k+2].mean(axis=1)**2 +
                      uReconstructY[timelev, :, k:k+2].mean(axis=1)**2)**0.5
        flux += speedLevel * thickness[timelev, :] * layerThicknessFractions[k]

    plt.scatter(xCell[iceIndices], yCell[iceIndices], markersize,
                c=np.array([[0.8, 0.8, 0.8], ]), marker=markershape,
                edgecolors='none')

    # add contours over the top
    contour_intervals = np.linspace(0.0, 20.0,  11)
    _contour_mpas(flux * 3600.0*24.0*365.0 / 10000.0, nCells, xCell, yCell,
                  contour_levs=contour_intervals)
    ax.set_aspect('equal')
    plt.title('Final flux (m$^2$ a$^{-1}$ / 10000)')
    plt.xlabel('X position (km)')
    plt.ylabel('Y position (km)')

    # ================
    # panel b - flow factor
    ax = fig.add_subplot(132, sharex=ax1, sharey=ax1)

    plt.scatter(xCell[iceIndices], yCell[iceIndices], markersize,
                c=np.array([[0.8, 0.8, 0.8], ]), marker=markershape,
                edgecolors='none')

    # add contours over the top
    # contour_intervals = np.linspace(0.0, 16.0, int(16.0/0.5)+1)

    # this is not used if FO velo solver is used
    if flwa[timelev, :, :].max() > 0.0:
        # NOT SURE WHICH LEVEL FLWA SHOULD COME FROM - so taking column average
        _contour_mpas(
            flwa[timelev, :, :].mean(axis=1) * 3600.0*24.0*365.0 / 1.0e-17,
            nCells, xCell, yCell)
    ax.set_aspect('equal')
    # Note: the paper's figure claims units of 10$^{-25}$ Pa$^{-3}$ a$^{-1}$
    # but the time unit appears to be 10^-17
    plt.title('Final flow factor (10$^{-17}$ Pa$^{-3}$ a$^{-1}$)')
    plt.xlabel('X position (km)')
    plt.ylabel('Y position (km)')

    if save_images:
        plt.savefig('EISMINT2-{}-steady.png'.format(experiment), dpi=150)

    # ================
    # DIVIDE EVOLUTION TIME SERIES
    # ================
    fig = plt.figure(3, facecolor='w')
    fig.suptitle('Payne et al. Fig. 5, 7, or 8', fontsize=10, fontweight='bold')

    # get indices for given time
    if experiment == 'b':
        endTime = 40000.0
    elif experiment == 'g':
        # WHL - Might change later to 80000
        endTime = 40000.0
    else:
        endTime = 80000.0

    # get index at divide - we set this up to be 750,750
    divideIndex = np.logical_and(xCell == 750.0, yCell == 750.0)

    # panel a - thickness
    fig.add_subplot(211)
    timeInd = np.nonzero(years <= endTime)[0][0:]
    plt.plot(years[timeInd]/1000.0, thickness[timeInd, divideIndex], 'k.-')
    plt.ylabel('Thickness (m)')

    # panel b - basal temperature
    fig.add_subplot(212)
    # skip the first index cause basalTemperature isn't calculated then
    timeInd = np.nonzero(years <= endTime)[0][1:]
    plt.plot(years[timeInd]/1000.0, basalTemperature[timeInd, divideIndex], 'k.-')
    plt.ylabel('Basal temperature (K)')
    plt.xlabel('Time (kyr)')

    if save_images:
        plt.savefig('EISMINT2-{}-divide.png'.format(experiment), dpi=150)

    # ================
    # TABLES
    # ================
    # Setup dictionaries of benchmark results for each experiment - values are
    # mean, min, max from Tables in Payne et al. 2000
    benchmarks = {'a': {'stattype': 'absolute',
                        'volume': (2.128, 2.060, 2.205),
                        'area': (1.034, 1.011, 1.097),
                        'meltfraction': (0.718, 0.587, 0.877),
                        'dividethickness': (3688.342, 3644.0, 3740.74),
                        'dividebasaltemp': (255.605, 254.16, 257.089)},
                  'b': {'stattype': 'relative',
                        'volume': (-2.589, -3.079, -2.132),
                        'area': (0.0, 0.0, 0.0),
                        'meltfraction': (11.836, 3.307, 21.976),
                        'dividethickness': (-4.927, -5.387, -4.071),
                        'dividebasaltemp': (4.623, 4.47, 4.988)},
                  'c': {'stattype': 'relative',
                        'volume': (-28.505, -29.226, -28.022),
                        'area': (-19.515, -20.369, -16.815),
                        'meltfraction': (-27.806, -39.353, -7.982),
                        'dividethickness': (-12.928, -13.948, -12.447),
                        'dividebasaltemp': (3.707, 3.389, 4.004)},
                  'd': {'stattype': 'relative',
                        'volume': (-12.085, -12.890, -11.654),
                        'area': (-9.489, -10.184, -6.924),
                        'meltfraction': (-1.613, -4.744, 1.001),
                        'dividethickness': (-2.181, -2.517, -1.985),
                        'dividebasaltemp': (-0.188, -0.209, -0.149)},
                  'f': {'stattype': 'absolute',
                        'volume': (0.0, 0.0, 0.0),
                        'area': (0.0, 0.0, 0.0),
                        'meltfraction': (0.0, 0.0, 0.0),
                        'dividethickness': (0.0, 0.0, 0.0),
                        'dividebasaltemp': (0.0, 0.0, 0.0)},
                  'g': {'stattype': 'absolute',
                        'volume': (1.589, 1.503, 2.205),
                        'area': (1.032, 1.016, 1.087),
                        'meltfraction': (0.352, 0.250, 0.780),
                        'dividethickness': (2365.206, 2212.550, 3681.431),
                        'dividebasaltemp': (249.134, 247.700, 255.381)}}

    # Get the benchmark dictionary
    bench = benchmarks[experiment]

    fig = plt.figure(4, facecolor='w')
    fig.suptitle('Payne et al. Table 4, 5, 6, 7, 8, or 9: showing '
                 'min/mean/max of community', fontsize=10, fontweight='bold')

    fig.add_subplot(151)
    volume = ((thickness[timelev, iceIndices] * areaCell[iceIndices]).sum()
              / 1000.0**3 / 10.0**6)
    # benchmark results
    plt.plot(np.zeros((3,)), bench['volume'], 'k*')
    if bench['stattype'] == 'relative':
        initIceIndices = np.where(thickness[0, :] > 0.0)[0]
        total_volume = \
            (thickness[0, initIceIndices] * areaCell[initIceIndices]).sum()
        volume = (volume / (total_volume / 1000.0**3 / 10.0**6) - 1.0) * 100.0
        plt.ylabel('Volume change (%)')
    else:
        plt.ylabel('Volume (10$^6$ km$^3$)')
    # MPAS results
    plt.plot((0.0,), volume, 'ro')
    plt.xticks(())
    logger.info("MALI volume = {}".format(volume))

    fig.add_subplot(152)
    area = (areaCell[iceIndices]).sum() / 1000.0**2 / 10.0**6
    areaAbsolute = area
    # benchmark results
    plt.plot(np.zeros((3,)), bench['area'], 'k*')
    if bench['stattype'] == 'relative':
        initArea = (areaCell[initIceIndices]).sum() / 1000.0**2 / 10.0**6
        area = (area / initArea - 1.0) * 100.0
        plt.ylabel('Area change (%)')
    else:
        plt.ylabel('Area (10$^6$ km$^2$)')
    # MPAS results
    plt.plot((0.0,), area, 'ro')
    plt.xticks(())
    logger.info("MALI area = {}".format(area))

    fig.add_subplot(153)
    # using threshold here to identify melted locations
    warmBedIndices = np.where(
        np.logical_and(thickness[timelev, :] > 0.0,
                       basalTemperature[timelev, :] >=
                       (basalPmpTemperature[timelev, :] - 0.01)))[0]
    meltfraction = (areaCell[warmBedIndices].sum() / 1000.0**2 / 10.0**6 /
                    areaAbsolute)
    # benchmark results
    plt.plot(np.zeros((3,)), bench['meltfraction'], 'k*')
    if bench['stattype'] == 'relative':
        # use time 1 instead of 0 since these fields aren't fully populated at
        # time 0
        initIceIndices = np.where(thickness[1, :] > 0.0)[0]
        initArea = (areaCell[initIceIndices].sum() / 1000.0**2 / 10.0**6)
        # using threshold here to identify melted locations
        initWarmBedIndices = \
            np.where(np.logical_and(thickness[1, :] > 0.0,
                                    basalTemperature[1, :] >=
                                    (basalPmpTemperature[1, :] - 0.01)))[0]
        initWarmArea = (areaCell[initWarmBedIndices].sum() / 1000.0**2 /
                        10.0**6)
        initMeltFraction = initWarmArea / initArea
        meltfraction = (meltfraction / initMeltFraction - 1.0) * 100.0
        plt.ylabel('Melt fraction change (%)')
    else:
        plt.ylabel('Melt fraction')
    # MPAS results
    plt.plot((0.0,), meltfraction, 'ro')
    plt.xticks(())
    logger.info("MALI melt fraction = {}".format(meltfraction))

    fig.add_subplot(154)
    dividethickness = thickness[timelev, divideIndex]
    # benchmark results
    plt.plot(np.zeros((3,)), bench['dividethickness'], 'k*')
    if bench['stattype'] == 'relative':
        dividethickness = \
            (dividethickness / thickness[0, divideIndex] - 1.0) * 100.0
        plt.ylabel('Divide thickness change (%)')
    else:
        plt.ylabel('Divide thickness (m)')
    plt.plot((0.0,), dividethickness, 'ro')  # MPAS results
    plt.xticks(())
    logger.info("MALI divide thickness = {}".format(dividethickness[0]))

    fig.add_subplot(155)
    dividebasaltemp = basalTemperature[timelev, divideIndex]
    # benchmark results
    plt.plot(np.zeros((3,)), bench['dividebasaltemp'], 'k*')
    if bench['stattype'] == 'relative':
        # use time 1 instead of 0 since these fields aren't fully populated at
        # time 0
        dividebasaltemp = dividebasaltemp - basalTemperature[1, divideIndex]
        plt.ylabel('Divide basal temp. change (K)')
    else:
        plt.ylabel('Divide basal temp. (K)')
    plt.plot((0.0,), dividebasaltemp, 'ro')  # MPAS results
    plt.xticks(())
    logger.info(
        "MALI divide basal temperature = {}".format(dividebasaltemp[0]))

    plt.tight_layout()

    plt.draw()
    if save_images:
        plt.savefig('EISMINT2-{}-table.png'.format(experiment), dpi=150)

    if hide_figs:
        logger.info("Plot display disabled with hide_plot config option.")
    else:
        plt.show()

    plt.close('all')


def _xtime_to_numtime(xtime):
    """
    Define a function to convert xtime character array to numeric time values
    using datetime objects
    """
    # First parse the xtime character array into a string

    # convert from the character array to an array of strings using the netCDF4
    # module's function
    xtimestr = netCDF4.chartostring(xtime)

    dt = []
    for stritem in xtimestr:
        # Get an array of strings that are Y,M,D,h,m,s
        itemarray = \
            stritem.strip().replace('_', '-').replace(':', '-').split('-')
        results = [int(i) for i in itemarray]
        # datetime has a bug where years less than 1900 are invalid on some
        # systems
        if results[0] < 1900:
            results[0] += 1900
        # * notation passes in the array as arguments
        dt.append(datetime.datetime(*results))

    # use the netCDF4 module's function for converting a datetime to a time
    # number
    numtime = netCDF4.date2num(dt, units='seconds since ' + str(dt[0]))
    return numtime


def _xtime_get_year(xtime):
    """
    Get an array of years from an xtime array, ignoring any partial year
    information
    """
    # First parse the xtime character array into a string

    # convert from the character array to an array of strings using the netCDF4
    # module's function
    xtimestr = netCDF4.chartostring(xtime)
    years = np.zeros((len(xtimestr),))
    for i in range(len(xtimestr)):
        # Get the year part and make it an integer
        years[i] = (int(xtimestr[i].split('-')[0]))
    return years


def _contour_mpas(field, nCells, xCell, yCell, contour_levs=None):
    """Contours irregular MPAS data on cells"""

    if contour_levs is None:
        contour_levs = np.array([0])

    # -- Now let's grid your data.
    # First we'll make a regular grid to interpolate onto.

    # may want to adjust the density of the regular grid
    numcols = int(nCells**0.5 * 4.0)
    numrows = numcols
    xc = np.linspace(xCell.min(), xCell.max(), numcols)
    yc = np.linspace(yCell.min(), yCell.max(), numrows)
    xi, yi = np.meshgrid(xc, yc)
    # -- Interpolate at the points in xi, yi
    zi = griddata((xCell, yCell), field, (xi, yi))
    # -- Display the results
    if len(contour_levs) == 1:
        im = plt.contour(xi, yi, zi)
    else:
        im = plt.contour(xi, yi, zi, contour_levs, cmap=plt.cm.jet)

    # to see the raw data on top
    # plt.scatter(xCell, yCell, c=temperature[timelev,:,-1], s=100,
    #             vmin=zi.min(), vmax=zi.max())
    plt.colorbar(im)
