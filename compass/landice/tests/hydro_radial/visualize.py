import numpy as np
import netCDF4
import matplotlib.pyplot as plt

from compass.step import Step


class Visualize(Step):
    """
    A step for visualizing the output from a dome test case
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

        self.add_input_file(filename='landice_grid.nc',
                            target='../{}/landice_grid.nc'.format(input_dir))

        self.add_input_file(filename='near_exact_solution_r_P_W.txt',
                            package='compass.landice.tests.hydro_radial')

    # depending on settings, this will produce no outputs, so we won't add any

    # no setup method is needed

    def run(self):
        """
        Run this step of the test case
        """
        visualize_hydro_radial(self.config, self.logger)


def visualize_hydro_radial(config, logger):
    """
    Plot the output from a hydro_radial test case

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step
    """
    section = config['hydro_radial_viz']

    time_slice = section.getint('time_slice')
    save_images = section.getboolean('save_images')
    hide_figs = section.getboolean('hide_figs')

    filename = 'output.nc'
    grid_filename = 'landice_grid.nc'

    f = netCDF4.Dataset(filename, 'r')
    xCell = f.variables['xCell'][:]
    yCell = f.variables['yCell'][:]
    xEdge = f.variables['xEdge'][:]
    yEdge = f.variables['yEdge'][:]
    h = f.variables['waterThickness'][time_slice, :]
    u = f.variables['waterVelocityCellX'][time_slice, :]
    P = f.variables['waterPressure'][time_slice, :]
    N = f.variables['effectivePressure'][time_slice, :]
    div = f.variables['divergence'][time_slice, :]
    opening = f.variables['openingRate'][time_slice, :]
    closing = f.variables['closingRate'][time_slice, :]
    melt = f.variables['basalMeltInput'][time_slice, :]
    sliding = f.variables['basalSpeed'][time_slice, :]
    days = f.variables['daysSinceStart'][:]

    logger.info("Total number of time levels={}".format(len(days)))
    logger.info("Using time slice {}, which is year {}".format(
        time_slice, days[time_slice] / 365.0))

    logger.info("Attempting to read thickness field from "
                "{}.".format(grid_filename))
    fin = netCDF4.Dataset(grid_filename, 'r')
    H = fin.variables['thickness'][0, :]

    # Find center row  - currently files are set up to have central row at y=0
    unique_ys = np.unique(yCell[:])
    centerY = unique_ys[len(unique_ys) // 2]
    logger.info("number of ys={}, center y index={}, center Y value={}".format(
        len(unique_ys), len(unique_ys) // 2, centerY))
    ind = np.nonzero(yCell[:] == centerY)
    x = xCell[ind] / 1000.0

    logger.info("start plotting.")

    fig = plt.figure(1, facecolor='w')

    # import exact solution
    fnameSoln = 'near_exact_solution_r_P_W.txt'
    soln = np.loadtxt(fnameSoln, delimiter=',')
    rsoln = soln[:, 0] / 1000.0
    Psoln = soln[:, 1] / 1.0e5
    Wsoln = soln[:, 2]

    # water thickness
    ax1 = fig.add_subplot(121)
    plt.plot(rsoln, Wsoln, 'k-', label='W exact')
    plt.plot(x, h[ind], 'r.--', label='W model')
    plt.xlabel('X-position (km)')
    plt.ylabel('water depth (m)')
    plt.legend()
    plt.plot([5.0, 5.0], [0.0, 1.0], ':k')
    plt.grid(True)

    # water pressure
    fig.add_subplot(122, sharex=ax1)
    plt.plot(x, H[ind] * 910.0 * 9.80616 / 1.0e5, 'g:', label='P_o')
    plt.plot(rsoln, Psoln, 'k-', label='P_w exact')
    plt.plot(x, P[ind] / 1.0e5, 'r.--', label='P_w model')
    plt.xlabel('X-position (km)')
    plt.ylabel('water pressure (bar)')
    plt.legend()
    plt.plot([5.0, 5.0], [0.0, 45.0], ':k')
    plt.grid(True)
    if save_images:
        plt.savefig('hydro_radial_vs_exact.png', dpi=150)

    # plot how close to SS we are
    fig = plt.figure(2, facecolor='w')
    ax1 = fig.add_subplot(211)
    for i in ind:
        plt.plot(days / 365.0, f.variables['waterThickness'][:, i])
    plt.xlabel('Years since start')
    plt.ylabel('water thickness (m)')
    plt.grid(True)

    fig.add_subplot(212, sharex=ax1)
    for i in ind:
        plt.plot(days / 365.0, f.variables['effectivePressure'][:, i] / 1.0e6)
    plt.xlabel('Years since start')
    plt.ylabel('effective pressure (MPa)')
    plt.grid(True)

    if save_images:
        plt.savefig('hydro_radial_steady_state.png', dpi=150)

    # plot opening/closing rates
    fig = plt.figure(3, facecolor='w')

    nplt = 5

    fig.add_subplot(nplt, 1, 1)
    plt.plot(x, opening[ind], 'r', label='opening')
    plt.plot(x, closing[ind], 'b', label='closing')
    plt.plot(x, melt[ind] / 1000.0, 'g', label='melt')
    plt.xlabel('X-position (km)')
    plt.ylabel('rate (m/s)')
    plt.legend()
    plt.grid(True)

    # SS N=f(h)
    fig.add_subplot(nplt, 1, 2)
    plt.plot(x, N[ind] / 1.0e6, '.-', label='modeled transient to SS')
    # steady state N=f(h) from the cavity evolution eqn
    N = (opening[ind] / (0.04 * 3.1709792e-24 * h[ind]))**0.3333333 / 1.0e6
    plt.plot(x, N, '.--r', label='SS N=f(h)')
    plt.xlabel('X-position (km)')
    plt.ylabel('effective pressure (MPa)')
    plt.grid(True)
    plt.legend()

    fig.add_subplot(nplt, 1, 3)
    plt.plot(x, u[ind])
    plt.ylabel('water velocity (m/s)')
    plt.grid(True)

    fig.add_subplot(nplt, 1, 4)
    plt.plot(x, u[ind] * h[ind])
    plt.ylabel('water flux (m2/s)')
    plt.grid(True)

    fig.add_subplot(nplt, 1, 5)
    plt.plot(x, div[ind])
    plt.plot(x, melt[ind] / 1000.0, 'g', label='melt')
    plt.ylabel('divergence (m/s)')
    plt.grid(True)

    if save_images:
        plt.savefig('hydro_radial_opening_closing.png', dpi=150)

    # plot some edge quantities
    inde = np.nonzero(yEdge[:] == centerY)
    xe = xEdge[inde] / 1000.0
    ve = f.variables['waterVelocity'][time_slice, :]
    dphie = f.variables['hydropotentialBaseSlopeNormal'][time_slice, :]
    he = f.variables['waterThicknessEdgeUpwind'][time_slice, :]
    fluxe = f.variables['waterFluxAdvec'][time_slice, :]

    fig = plt.figure(5, facecolor='w')
    nplt = 5

    ax1 = fig.add_subplot(nplt, 1, 1)
    plt.plot(xe, dphie[inde], '.')
    plt.ylabel('dphidx edge)')
    plt.grid(True)

    fig.add_subplot(nplt, 1, 2, sharex=ax1)
    plt.plot(x, P[ind], 'x')
    plt.ylabel('dphidx edge)')
    plt.grid(True)

    fig.add_subplot(nplt, 1, 3, sharex=ax1)
    plt.plot(xe, ve[inde], '.')
    plt.ylabel('vel edge)')
    plt.grid(True)

    fig.add_subplot(nplt, 1, 4, sharex=ax1)
    plt.plot(xe, he[inde], '.')
    plt.plot(x, h[ind], 'x')
    plt.ylabel('h edge)')
    plt.grid(True)

    fig.add_subplot(nplt, 1, 5, sharex=ax1)
    plt.plot(xe, fluxe[inde], '.')
    plt.ylabel('flux edge)')
    plt.grid(True)

    # ==========
    # Make plot similar to Bueler and van Pelt Fig. 5

    # get thickness/pressure at time 0 - this should be the nearly-exact
    # solution interpolated onto the MPAS mesh
    h0 = f.variables['waterThickness'][0, :]
    P0 = f.variables['waterPressure'][0, :]
    # assuming sliding has been zeroed where there is no ice, so we don't need
    # to get the thickness field
    hasice = sliding > 0.0

    Werr = np.absolute(h - h0)
    Perr = np.absolute(P - P0)
    dcEdge = f.variables['dcEdge'][:]
    # ideally should restrict this to edges with ice
    dx = dcEdge.mean()

    if save_images:
        plt.savefig('hydro_radial_edge.png', dpi=150)

    fig = plt.figure(6, facecolor='w')

    ax = fig.add_subplot(2, 1, 1)
    plt.plot(dx, Werr[hasice].mean(), 's', label='avg W err')
    plt.plot(dx, Werr[hasice].max(), 'x', label='max W err')
    ax.set_yscale('log')
    plt.grid(True)
    plt.legend()
    plt.xlabel('delta x (m)')
    plt.ylabel('error in W (m)')
    logger.info("avg W err={}".format(Werr[hasice].mean()))
    logger.info("max W err={}".format(Werr[hasice].max()))

    ax = fig.add_subplot(2, 1, 2)
    plt.plot(dx, Perr[hasice].mean() / 1.0e5, 's', label='avg P err')
    plt.plot(dx, Perr[hasice].max() / 1.0e5, 'x', label='max P err')
    ax.set_yscale('log')
    plt.grid(True)
    plt.legend()
    plt.xlabel('delta x (m)')
    plt.ylabel('error in P (bar)')
    logger.info("avg P err={}".format(Perr[hasice].mean() / 1.0e5))
    logger.info("max P err={}".format(Perr[hasice].max() / 1.0e5))

    logger.info("plotting complete")

    plt.draw()
    if save_images:
        plt.savefig('hydro_radial_error.png', dpi=150)

    if hide_figs:
        logger.info("Plot display disabled with hide_plot config option.")
    else:
        plt.show()

    f.close()
