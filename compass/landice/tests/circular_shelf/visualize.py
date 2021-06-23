import netCDF4
import matplotlib.pyplot as plt
import numpy as np

from compass.step import Step


class Visualize(Step):
    """
    A step for visualizing the output from a circular_shelf test case

    Attributes
    ----------
    mesh_type : str
        The resolution or mesh type of the test case
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
        visualize_circular_shelf(self.config, self.logger,
                                 filename='output.nc')


def visualize_circular_shelf(config, logger, filename):
    """
    Plot the output from a circular_shelf test case

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step

    filename : str
        file to visualize
    """
    section = config['circular_shelf_viz']

    time_slice = section.getint('time_slice')
    save_images = section.getboolean('save_images')
    hide_figs = section.getboolean('hide_figs')

    # Note: this may be slightly wrong for some calendar types!
    secInYr = 3600.0 * 24.0 * 365.0

    f = netCDF4.Dataset(filename, 'r')

    thickness = f.variables['thickness']
    if 'bedTopography' in f.variables:
        bedTopography = f.variables['bedTopography']  # not needed
    else:
        logger.write("bedTopography not in file.  Continuing without it.")
    xCell = f.variables['xCell'][:] / 1000.0
    yCell = f.variables['yCell'][:] / 1000.0
    lowerSurface = f.variables['lowerSurface']
    upperSurface = f.variables['upperSurface']
    uReconstructX = f.variables['uReconstructX']
    uReconstructY = f.variables['uReconstructY']
    layerInterfaceSigma = f.variables['layerInterfaceSigma'][:]

    vert_levs = len(f.dimensions['nVertLevels'])

    velnorm = (uReconstructX[:]**2 + uReconstructY[:]**2)**0.5 * secInYr
    logger.info("Maximum velocity (m/yr) at cell centers in domain: {}"
                .format(velnorm.max()))

    ##################
    # FIGURE: Map view surface and bed velocity
    ##################
    fig = plt.figure(1)
    fig.add_subplot(121, aspect='equal')
    plt.scatter(xCell[:], yCell[:], 80, velnorm[time_slice, :, 0], marker='h',
                edgecolors='none')
    plt.colorbar()
    plt.quiver(xCell[:], yCell[:], uReconstructX[time_slice, :, 0] * secInYr,
               uReconstructY[time_slice, :, 0] * secInYr)
    plt.title('surface speed (m/yr)')
    plt.draw()
    fig.add_subplot(122, aspect='equal')
    plt.scatter(xCell[:], yCell[:], 80, velnorm[time_slice, :, -1], marker='h',
                edgecolors='none')
    plt.colorbar()
    plt.quiver(xCell[:], yCell[:], uReconstructX[time_slice, :, -1] * secInYr,
               uReconstructY[time_slice, :, -1] * secInYr)
    plt.title('basal speed (m/yr)')
    plt.draw()
    if save_images:
        plt.savefig('circ_shelf_velos.png')

    ##################
    # FIGURE: Cross-section of surface velocity through centerline
    ##################
    fig = plt.figure(2)
    # find cells on a longitudinal cross-section at y=0
    # Note this may not work on a variable resolution mesh
    # Note this assumes the setup script put the center of a cell at the
    # center of the mesh at 0,0.
    indXsect = np.where(yCell == 0.0)[0]
    indXsectIce = np.where(np.logical_and(yCell == 0.0,
                                          thickness[time_slice, :] > 0.0))[0]

    # contour speed across the cross-section
    plt.contourf(np.tile(xCell[indXsectIce], (vert_levs+1,1)).transpose(),
        (np.tile(thickness[time_slice, indXsectIce], (vert_levs+1,1)) *
        np.tile(layerInterfaceSigma, (len(indXsectIce),1)).transpose() +
        lowerSurface[time_slice, indXsectIce]).transpose(),
        velnorm[time_slice,indXsectIce,:], 100)
    # plot x's at the velocity data locations
    plt.plot(np.tile(xCell[indXsectIce], (vert_levs+1,1)).transpose(),
        (np.tile(thickness[time_slice, indXsectIce], (vert_levs+1,1)) *
        np.tile(layerInterfaceSigma, (len(indXsectIce),1)).transpose() +
        lowerSurface[time_slice, indXsectIce]).transpose(),
        'kx')#, label='velocity points')
    cbar=plt.colorbar()
    cbar.set_label('speed (m/yr)', rotation=270)

    plt.plot(xCell[indXsectIce], upperSurface[time_slice, indXsectIce], 'ro-',
             label="Upper surface")
    plt.plot(xCell[indXsectIce], lowerSurface[time_slice, indXsectIce], 'bo-',
             label="Lower surface")
    try:
        plt.plot(xCell[indXsect], bedTopography[time_slice, indXsect], 'go-',
                 label="Bed topography")
    except:
        logger.write("Skipping plotting of bedTopography.")
    plt.plot(xCell[indXsect], xCell[indXsect] * 0.0, ':k', label="sea level")
    plt.legend(loc='best')
    plt.title('cross-section at y=0' )
    plt.draw()
    if save_images:
        plt.savefig('circ_shelf_xsect.png')


    ##################
    # FIGURE: scatter plot of upper and lower surfaces
    ##################
    #fig = plt.figure(3)
    #ax = fig.add_subplot(121, aspect='equal')
    #plt.scatter(xCell[:], yCell[:], 80, upperSurface[time_slice,:], marker='h', edgecolors='none')
    #plt.colorbar()
    #plt.title('upper surface (m)' )
    #plt.draw()
    #ax = fig.add_subplot(122, aspect='equal')
    #plt.scatter(xCell[:], yCell[:], 80, lowerSurface[time_slice,:], marker='h', edgecolors='none')
    #plt.colorbar()
    #plt.title('lower surface (m)' )
    #plt.draw()
    #if options.save_images:
    #        plt.savefig('circ_shelf_surfaces.png')

    if not hide_figs:
        plt.show()

    f.close()
