import matplotlib.pyplot as plt
import netCDF4
import numpy

from compass.step import Step


class Visualize(Step):
    """
    A step for visualizing the output from a dome test case

    Attributes
    ----------
    """
    def __init__(self, test_case, name='visualize', subdir=None,
                 input_dir='run_model'):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        name : str, optional
            the name of the test case

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        input_dir : str, optional
            The input directory within the test case with a file ``output.nc``
            to visualize
        """
        super().__init__(test_case=test_case, name=name, subdir=subdir)

        self.add_input_file(filename='output.nc',
                            target='../{}/output.nc'.format(input_dir))

        # depending on settings, this may produce no outputs, so we won't add
        # any

    # no setup method is needed

    def run(self):
        """
        Run this step of the test case
        """
        visualize_slm_circsheet(self.config, self.logger, filename='output.nc')


def visualize_slm_circsheet(config, logger, filename):
    """
    Plot the output from a dome test case

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step

    filename : str
        file to visualize
    """
    section = config['dome_viz']

    time_slice = section.getint('time_slice')
    save_images = section.getboolean('save_images')
    hide_figs = section.getboolean('hide_figs')

    # Note: this may be slightly wrong for some calendar types!
    secInYr = 3600.0 * 24.0 * 365.0

    f = netCDF4.Dataset(filename, 'r')

    times = f.variables['xtime']
    thickness = f.variables['thickness']
    # dcEdge = f.variables['dcEdge']
    # bedTopography = f.variables['bedTopography']  # not needed
    xCell = f.variables['xCell']
    yCell = f.variables['yCell']
    xEdge = f.variables['xEdge']
    yEdge = f.variables['yEdge']
    angleEdge = f.variables['angleEdge']
    temperature = f.variables['temperature']
    lowerSurface = f.variables['lowerSurface']
    upperSurface = f.variables['upperSurface']
    normalVelocity = f.variables['normalVelocity']
    # uReconstructX = f.variables['uReconstructX']
    uReconstructX = f.variables['uReconstructX']
    uReconstructY = f.variables['uReconstructY']

    vert_levs = len(f.dimensions['nVertLevels'])

    time_length = times.shape[0]

    logger.info("vert_levs = {};  time_length = {}".format(vert_levs,
                                                           time_length))

    var_slice = thickness[time_slice, :]

    fig = plt.figure(1, facecolor='w')
    fig.add_subplot(111, aspect='equal')
    # C = plt.contourf(xCell, yCell, var_slice )
    plt.scatter(xCell[:], yCell[:], 80, var_slice, marker='h',
                edgecolors='none')
    plt.colorbar()
    plt.title('thickness at time {}'.format(time_slice))
    plt.draw()
    if save_images:
        logger.info("Saving figures to files.")
        plt.savefig('dome_thickness.png')

    fig = plt.figure(2)
    fig.add_subplot(121, aspect='equal')
    plt.scatter(xCell[:], yCell[:], 80, lowerSurface[time_slice, :],
                marker='h', edgecolors='none')
    plt.colorbar()
    plt.title('lower surface at time {}'.format(time_slice))
    plt.draw()
    fig.add_subplot(122, aspect='equal')
    plt.scatter(xCell[:], yCell[:], 80, upperSurface[time_slice, :],
                marker='h', edgecolors='none')
    plt.colorbar()
    plt.title('upper surface at time {}'.format(time_slice))
    plt.draw()
    if save_images:
        plt.savefig('dome_surfaces.png')

    fig = plt.figure(3)
    for templevel in range(0, vert_levs):
        fig.add_subplot(3, 4, templevel + 1, aspect='equal')
        var_slice = temperature[time_slice, :, templevel]
        # C = plt.contourf(xCell, yCell, var_slice )
        plt.scatter(xCell[:], yCell[:], 40, var_slice, marker='h',
                    edgecolors='none')
        plt.colorbar()
        plt.title('temperature at level {} at time {}'.format(templevel,
                                                              time_slice))
        plt.draw()
    if save_images:
        plt.savefig('dome_temperature.png')

    fig = plt.figure(4)
    fig.add_subplot(121, aspect='equal')
    plt.scatter(xEdge[:], yEdge[:], 80,
                normalVelocity[time_slice, :, vert_levs - 1] * secInYr,
                marker='h', edgecolors='none')
    plt.colorbar()
    normalVel = normalVelocity[time_slice, :, vert_levs - 1]
    plt.quiver(xEdge[:], yEdge[:],
               numpy.cos(angleEdge[:]) * normalVel * secInYr,
               numpy.sin(angleEdge[:]) * normalVel * secInYr)
    plt.title('normalVelocity of bottom layer at time {}'.format(time_slice))
    plt.draw()
    fig.add_subplot(122, aspect='equal')
    plt.scatter(xEdge[:], yEdge[:], 80,
                normalVelocity[time_slice, :, 0] * secInYr, marker='h',
                edgecolors='none')
    plt.colorbar()
    normalVel = normalVelocity[time_slice, :, 0]
    plt.quiver(xEdge[:], yEdge[:],
               numpy.cos(angleEdge[:]) * normalVel * secInYr,
               numpy.sin(angleEdge[:]) * normalVel * secInYr)
    plt.title('normalVelocity of top layer at time {}'.format(time_slice))
    plt.draw()
    if save_images:
        plt.savefig('dome_normalVelocity.png')

    fig = plt.figure(5, facecolor='w')
    fig.add_subplot(121, aspect='equal')
    plt.scatter(xCell[:], yCell[:], 80,
                uReconstructX[time_slice, :, 0] * secInYr, marker='h',
                edgecolors='none')
    plt.colorbar()
    plt.quiver(xCell[:], yCell[:], uReconstructX[time_slice, :, 0] * secInYr,
               uReconstructY[time_slice, :, 0] * secInYr)
    plt.title('uReconstructX of top layer at time {}'.format(time_slice))
    plt.draw()
    fig.add_subplot(122, aspect='equal')
    plt.scatter(xCell[:], yCell[:], 80,
                uReconstructY[time_slice, :, 0] * secInYr, marker='h',
                edgecolors='none')
    plt.colorbar()
    plt.quiver(xCell[:], yCell[:], uReconstructX[time_slice, :, 0] * secInYr,
               uReconstructY[time_slice, :, 0] * secInYr)
    plt.title('uReconstructY of top layer at time {}'.format(time_slice))
    plt.draw()
    if save_images:
        plt.savefig('dome_uReconstruct.png')

    if hide_figs:
        logger.info("Plot display disabled with hide_plot config option.")
    else:
        plt.show()

    f.close()
