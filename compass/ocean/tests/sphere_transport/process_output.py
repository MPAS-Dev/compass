import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
import matplotlib
import numpy as np
from importlib import resources


def appx_mesh_size(dataset):
    ncells = len(dataset.dimensions["nCells"])
    return np.sqrt(4 * np.pi / ncells)


def compute_error_from_output_ncfile(dataset, lev=1):
    """
    Given a netCDF4 Dataset associated with the output.nc file from a test
    case in the sphere_transport test group, this function computes the
    linf and l2 relative error values by comparing the final time step to
    the initial condition.

    Parameters
    ----------
    dataset : NetCDF4.Dataset
        a dataset initialized with an MPAS output.nc file.

    lev: int, optional
        vertical level to plot.

    Returns
    -------
    result : dict
        a dictionary containing the linf and l2 relative errors for each of the
        3 debug tracers.
    """
    tracer1_exact = dataset.variables["tracer1"][0, :, lev]
    tracer2_exact = dataset.variables["tracer2"][0, :, lev]
    tracer3_exact = dataset.variables["tracer3"][0, :, lev]
    tracer1_error = np.abs(
        dataset.variables["tracer1"][12, :, lev] - tracer1_exact)
    tracer2_error = np.abs(
        dataset.variables["tracer2"][12, :, lev] - tracer2_exact)
    tracer3_error = np.abs(
        dataset.variables["tracer3"][12, :, lev] - tracer3_exact)
    tracer1_max = np.amax(tracer1_exact)
    tracer2_max = np.amax(tracer2_exact)
    tracer3_max = np.amax(tracer3_exact)
    tracer1_min = np.amin(tracer1_exact)
    tracer2_min = np.amin(tracer2_exact)
    tracer3_min = np.amin(tracer3_exact)
    cell_area = dataset.variables["areaCell"][:]
    tracer1_linf = np.amax(tracer1_error) / np.amax(np.abs(tracer1_exact))
    tracer2_linf = np.amax(tracer2_error) / np.amax(np.abs(tracer2_exact))
    tracer3_linf = np.amax(tracer3_error) / np.amax(np.abs(tracer3_exact))
    tracer1_l2 = np.sqrt(
        np.sum(
            np.square(tracer1_error) *
            cell_area) /
        np.sum(
            np.square(tracer1_exact) *
            cell_area))
    tracer2_l2 = np.sqrt(
        np.sum(
            np.square(tracer2_error) *
            cell_area) /
        np.sum(
            np.square(tracer2_exact) *
            cell_area))
    tracer3_l2 = np.sqrt(
        np.sum(
            np.square(tracer3_error) *
            cell_area) /
        np.sum(
            np.square(tracer3_exact) *
            cell_area))
    tracer1_mass0 = np.sum(cell_area * tracer1_exact)
    tracer2_mass0 = np.sum(cell_area * tracer2_exact)
    tracer3_mass0 = np.sum(cell_area * tracer3_exact)
    over1 = []
    under1 = []
    over2 = []
    under2 = []
    over3 = []
    under3 = []
    if len(dataset.dimensions["Time"]) == 13:
        for i in range(13):
            dmax1 = dataset.variables["tracer1"][i, :, lev] - tracer1_max
            dmax2 = dataset.variables["tracer2"][i, :, lev] - tracer2_max
            dmax3 = dataset.variables["tracer3"][i, :, lev] - tracer3_max
            dmin1 = dataset.variables["tracer1"][i, :, lev] - tracer1_min
            dmin2 = dataset.variables["tracer2"][i, :, lev] - tracer2_min
            dmin3 = dataset.variables["tracer3"][i, :, lev] - tracer3_min
            isover1 = dmax1 > 0
            isunder1 = dmin1 < 0
            isover2 = dmax2 > 0
            isunder2 = dmin2 < 0
            isover3 = dmax3 > 0
            isunder3 = dmin3 < 0
            over1.append(np.amax(dmax1 * isover1) /
                         (tracer1_max - tracer1_min))
            under1.append(np.amax(-dmin1 * isunder1) /
                          (tracer1_max - tracer1_min))
            over2.append(np.amax(dmax2 * isover2) /
                         (tracer2_max - tracer2_min))
            under2.append(np.amax(-dmin2 * isunder2) /
                          (tracer2_max - tracer2_min))
            over3.append(np.amax(dmax3 * isover3) /
                         (tracer3_max - tracer3_min))
            under3.append(np.amax(-dmin3 * isunder3) /
                          (tracer3_max - tracer3_min))
    else:
        over1 = 0
        over2 = 0
        over3 = 0
        under1 = 0
        under2 = 0
        under3 = 0
    tracer1_mass12 = np.sum(cell_area *
                            dataset.variables["tracer1"][12, :, lev])
    tracer2_mass12 = np.sum(cell_area *
                            dataset.variables["tracer2"][12, :, lev])
    tracer3_mass12 = np.sum(cell_area *
                            dataset.variables["tracer3"][12, :, lev])
    tracer1_masserr = np.abs(tracer1_mass0 - tracer1_mass12) / tracer1_mass0
    tracer2_masserr = np.abs(tracer2_mass0 - tracer2_mass12) / tracer2_mass0
    tracer3_masserr = np.abs(tracer3_mass0 - tracer3_mass12) / tracer3_mass0
    filament_tau = np.linspace(0, 1, 21)
    filament_area = np.zeros(21)
    filament_area0 = np.ones(21)
    for i, tau in enumerate(filament_tau):
        cells_above_tau = dataset.variables["tracer2"][6, :, lev] >= tau
        cells_above_tau0 = dataset.variables["tracer2"][0, :, lev] >= tau
        filament_area[i] = np.sum(cell_area * cells_above_tau)
        filament_area0[i] = np.sum(cells_above_tau0 * cell_area)
    filament_norm = filament_area / filament_area0

    result = dict()
    result["tracer1"] = {
        "linf": tracer1_linf,
        "l2": tracer1_l2,
        "over": over1,
        "under": under1,
        "mass": tracer1_masserr}
    result["tracer2"] = {
        "linf": tracer2_linf,
        "l2": tracer2_l2,
        "over": over2,
        "under": under2,
        "filament": filament_norm,
        "mass": tracer2_masserr}
    result["tracer3"] = {
        "linf": tracer3_linf,
        "l2": tracer3_l2,
        "over": over3,
        "under": under3,
        "mass": tracer3_masserr}
    return result


def compute_convergence_rates(dlambda, linf, l2):
    """
    Given a set of approximate mesh sizes (dlambda) and the corresponding
    linf and l2 relative error values, this function computes the
    approximate convergence rates for each error.  These values are
    computed by compute_error_from_output_ncfile for tracer1.

    Parameters
    ----------
    dlambda : numpy.ndarray
        approximate mesh size

    linf : numpy.ndarray
        linf relative error associated with each mesh size

    l2: numpy.ndarray
        l2 relative error associated with each mesh size

    Returns
    -------
    linfrates, l2rates : numpy.ndarray
        Approximate convergence rates for each error.
    """
    runs = np.log(dlambda[1:]) - np.log(dlambda[:-1])
    linfrises = np.log(linf[1:]) - np.log(linf[:-1])
    l2rises = np.log(l2[1:]) - np.log(l2[:-1])
    return linfrises / runs, l2rises / runs


def print_error_conv_table(
        tcname,
        resvals,
        dlambda,
        l2,
        l2rates,
        linf,
        linfrates):
    """
    Print error values and approximate convergence rates to the console
    as a table.

    Parameters
    ----------
    tcname : str
        Name of test case

    resvals : list
        resolution values such as 240, for ``QU240``

    dlambda : numpy.ndarray
        approximate mesh size

    l2 : numpy.ndarray
        l2 error, computed by ``compute_error_from_output_ncfile()``
        for ``tracer1``
    l2rates : numpy.ndarray
        appx. convergence rates for l2, computed by
        ``compute_convergence_rates()``
    linf : numpy.ndarray
        linf error, computed by ``compute_error_from_output_ncfile()``
        for ``tracer1``

    linfrates : numpy.ndarray
        appx. convergence rates for linf, computed
        by ``compute_convergence_rates()``
    """
    table_rows = []
    for i, r in enumerate(resvals):
        table_rows.append([r,
                           dlambda[i],
                           l2[i],
                           l2rates[i - 1] if i > 0 else '-',
                           linf[i],
                           linfrates[i - 1] if i > 0 else '-'])
    print(tcname + ": error data for tracer1")
    row_headers = ["res", "dlambda", "l2", "l2 rate", "linf", "linf rate"]
    row_format = "{:>24}" * len(row_headers)
    print(row_format.format(*row_headers))
    for row in table_rows:
        print(row_format.format(*row))


def read_ncl_rgb_file(cmap_filename):
    """
    Read a .rgb file from the NCAR Command Language, and return a
    matplotlib colormap.

    Prerequisite: Download an RGB file using the links provided by
    the NCL web pages,
    https://www.ncl.ucar.edu/Document/Graphics/color_table_gallery.shtml

    Parameters
    ----------
    cmap_filename : str
        downloaded .rgb file name

    Returns
    -------
    result : matplotlib.Colormap
        colormap usable by matplotlib that matches the ncl colormap
    """
    map_file_found = False
    try:
        with resources.open_text(
                "compass.ocean.tests.sphere_transport.resources", cmap_filename) \
                as f:
            flines = f.readlines()
        map_file_found = True
    except BaseException:
        pass
    if map_file_found:
        ncolors = int(flines[0].split()[-1])
        rgb = np.zeros((ncolors, 3))
        for i, l in enumerate(flines[3:]):
            ls = l.split()
            for j in range(3):
                rgb[i, j] = ls[j]
        rgb /= 255
        result = ListedColormap(rgb, name=cmap_filename)
    else:
        print("error reading ncl colormap. using matplotlib default instead.")
        result = matplotlib.cm.get_cmap()
    return result


def plot_sol(fig, tcname, dataset):
    """
    Plot the solution at time 0, t = T/2, and T=T for test cases in the
    ``sphere_transport`` test group. Each tracer is plotted in its own row.
    Columns correspond to t=0, t=T/2, and t=T.

    Parameters
    ----------
    fig : matplotlib.Figure
        A matplotlib figure instance

    tcname : str
        name of the test case whose solutions will be plotted

    dataset : NetCDF4.Dataset
        instance of a netCDF4 dataset initialized initialized with
        an MPAS output.nc file.
    """
    xc = dataset.variables["lonCell"][:]
    yc = dataset.variables["latCell"][:]
    gspc = GridSpec(nrows=4, ncols=3, figure=fig)
    yticks = np.pi * np.array([-0.5, -0.25, 0, 0.25, 0.5])
    yticklabels = [-90, -45, 0, 45, 90]
    xticks = np.pi * np.array([0, 0.5, 1, 1.5, 2])
    xticklabels = [0, 90, 180, 270, 360]

    clev = np.linspace(0, 1.1, 21)
    diffmin = -0.25
    diffmax = -diffmin
    dlev = np.linspace(diffmin, diffmax, 21)
    nclcmap = read_ncl_rgb_file("wh-bl-gr-ye-re.rgb")
    axes = []
    for i in range(4):
        for j in range(3):
            axes.append(fig.add_subplot(gspc[i, j]))
    axes[0].tricontourf(xc, yc, dataset.variables["tracer1"]
                        [0, :, 1], levels=clev, cmap=nclcmap,
                        vmin=0, vmax=1.1)
    axes[0].set_title('sol. t=0')
    axes[0].set_ylabel('tracer 1')
    axes[1].tricontourf(xc, yc, dataset.variables["tracer1"]
                        [6, :, 1], levels=clev, cmap=nclcmap,
                        vmin=0, vmax=1.1)
    axes[1].set_title('sol. t=T/2')
    axes[2].tricontourf(xc,
                        yc,
                        dataset.variables["tracer1"][12, :, 1] -
                        dataset.variables["tracer1"][0, :, 1],
                        levels=dlev,
                        cmap="seismic",
                        vmin=diffmin,
                        vmax=diffmax)
    axes[2].set_title('err. t=T')
    axes[3].tricontourf(xc, yc, dataset.variables["tracer2"]
                        [0, :, 1], levels=clev, cmap=nclcmap,
                        vmin=0, vmax=1.1)
    axes[3].set_ylabel('tracer 2')
    axes[4].tricontourf(xc, yc, dataset.variables["tracer2"]
                        [6, :, 1], levels=clev, cmap=nclcmap,
                        vmin=0, vmax=1.1)
    axes[5].tricontourf(xc,
                        yc,
                        dataset.variables["tracer2"][12, :, 1] -
                        dataset.variables["tracer2"][0, :, 1],
                        levels=dlev,
                        cmap="seismic",
                        vmin=diffmin,
                        vmax=diffmax)
    tcm = axes[6].tricontourf(xc, yc, dataset.variables["tracer3"]
                              [0, :, 1], levels=clev, cmap=nclcmap,
                              vmin=0, vmax=1.1)
    axes[6].set_ylabel('tracer 3')
    axes[7].tricontourf(xc, yc, dataset.variables["tracer3"]
                        [6, :, 1], levels=clev, cmap=nclcmap,
                        vmin=0, vmax=1.1)
    cm = axes[8].tricontourf(xc,
                             yc,
                             dataset.variables["tracer3"][12, :, 1] -
                             dataset.variables["tracer3"][0, :, 1],
                             levels=dlev,
                             cmap="seismic",
                             vmin=diffmin,
                             vmax=diffmax)
    lcm = axes[9].tricontourf(xc, yc, dataset.variables["layerThickness"]
                              [0, :, 1])
    axes[9].set_ylabel('layer thickness')
    axes[10].tricontourf(xc, yc, dataset.variables["layerThickness"]
                         [0, :, 1])
    axes[11].tricontourf(xc,
                         yc,
                         dataset.variables["layerThickness"][12, :, 1] -
                         dataset.variables["layerThickness"][0, :, 1],
                         levels=dlev,
                         cmap="seismic",
                         vmin=diffmin,
                         vmax=diffmax)

    for i in range(12):
        axes[i].set_xticks(xticks)
        axes[i].set_yticks(yticks)
        if i % 3 != 0:
            axes[i].set_yticklabels([])
    for i in range(4):
        axes[3 * i].set_yticklabels(yticklabels)
    for i in range(3):
        axes[9 + i].set_xticklabels(xticklabels)
    for i in range(9):
        axes[i].set_xticklabels([])
    cb1 = fig.colorbar(cm, ax=axes[8])
    cb2 = fig.colorbar(tcm, ax=axes[5])
    # cb3 = fig.colorbar(lcm, ax=axes[11])
    fig.suptitle(tcname)


def make_convergence_arrays(tcdata):
    """
    Collects data from a set of test case runs at different resolutions
    to use for convergence data analysis and plotting.

    Parameters
    ----------
    tcdata : dict
        a dictionary whose keys are the resolution values for a
        ``sphere_transport`` test case

    Returns
    -------
    dlambda : list
        an array of increasing appx. mesh sizes

    linf1 : list
        the linf error of tracer1 for each resolution/mesh size pair

    linf2 : list
        the linf error of tracer2 for each resolution/mesh size pair

    linf3 : list
        the linf error of tracer3 for each resolution/mesh size pair

    l21 : list
        the l2 error of tracer1 for each resolution/mesh size pair

    l22 : list
        the l2 error of tracer2 for each resolution/mesh size pair

    l23 : list
        the l2 error of tracer3 for each resolution/mesh size pair

    filament : list
        the "filament norm" for tracer2 at t=T/2.
        See Sec. 3.3 of LSPT2012.
    """
    rvals = sorted(tcdata.keys())
    rvals.reverse()
    dlambda = []
    linf1 = []
    linf2 = []
    linf3 = []
    l21 = []
    l22 = []
    l23 = []
    filament = []
    u1 = []
    o1 = []
    u2 = []
    o2 = []
    u3 = []
    o3 = []
    mass1 = []
    mass2 = []
    mass3 = []
    for r in rvals:
        dlambda.append(tcdata[r]['appx_mesh_size'])
        linf1.append(tcdata[r]['err']['tracer1']['linf'])
        linf2.append(tcdata[r]['err']['tracer2']['linf'])
        linf3.append(tcdata[r]['err']['tracer3']['linf'])
        l21.append(tcdata[r]['err']['tracer1']['l2'])
        l22.append(tcdata[r]['err']['tracer2']['l2'])
        l23.append(tcdata[r]['err']['tracer3']['l2'])
        filament.append(tcdata[r]['err']['tracer2']['filament'])
        u1.append(np.array(tcdata[r]['err']['tracer1']['under']))
        o1.append(np.array(tcdata[r]['err']['tracer1']['over']))
        u2.append(np.array(tcdata[r]['err']['tracer2']['under']))
        o2.append(np.array(tcdata[r]['err']['tracer2']['over']))
        u3.append(np.array(tcdata[r]['err']['tracer3']['under']))
        o3.append(np.array(tcdata[r]['err']['tracer3']['over']))
        mass1.append(tcdata[r]['err']['tracer1']['mass'])
        mass2.append(tcdata[r]['err']['tracer2']['mass'])
        mass3.append(tcdata[r]['err']['tracer3']['mass'])
    return dlambda, linf1, linf2, linf3, l21, l22, l23, filament, u1, o1, \
        u2, o2, u3, o3, mass1, mass2, mass3


def print_data_as_csv(tcname, tcdata):
    """
    Print test case data in csv format

    Parameters
    ----------
    tcname : str
        name of the test case whose solutions will be plotted

    tcdata : dict
        a dictionary whose keys are the resolution values for a
        ``sphere_transport`` test case
    """
    rvals = sorted(tcdata.keys())
    rvals.reverse()
    dlambda, linf1, linf2, linf3, l21, l22, l23, _, u1, o1, u2, o2, u3, o3, \
        mass1, mass2, mass3 = make_convergence_arrays(tcdata)
    headers = [
        "res",
        "dlambda",
        "linf1",
        "linf2",
        "linf3",
        "l21",
        "l22",
        "l23",
        "under1",
        "over1",
        "under2",
        "over2",
        "under3",
        "over3",
        "mass1",
        "mass2",
        "mass3"]
    print('-----------')
    print(' csv follows ')
    print('-----------')
    print('')
    print(",".join(headers))
    for i, r in enumerate(rvals):
        print(",".join([str(r),
                        str(dlambda[i]),
                        str(linf1[i]),
                        str(linf2[i]),
                        str(linf3[i]),
                        str(l21[i]),
                        str(l22[i]),
                        str(l23[i]),
                        str(np.amax(np.abs(u1[i]))),
                        str(np.amax(np.abs(o1[i]))),
                        str(np.amax(np.abs(u2[i]))),
                        str(np.amax(np.abs(o2[i]))),
                        str(np.amax(np.abs(u3[i]))),
                        str(np.amax(np.abs(o3[i]))),
                        str(mass1[i]),
                        str(mass2[i]),
                        str(mass3[i])]))
    print('')
    print('-----------')


def plot_convergence(
        ax,
        tcname,
        dlambda,
        resvals,
        linf1,
        l21,
        linf2,
        l22,
        linf3,
        l23):
    """
    Creates a convergence plot for a test case from the ``sphere_transport``
    test group.

    Parameters
    ----------
    ax : matplotlib.Axes
        A matplotlib Axes instance

    tcname : str
        The name of the test case

    dlambda : numpy.ndarray
        An array of mesh size values

    resvals : numpy.ndarray
        An integer array of resolution values, e.g., [120, 240]
        for ``QU120`` and ``QU240``

    linf1 : numpy.ndarray
        the linf error for tracer1

    l21 : numpy.ndarray
        the l2 error for tracer1

    linf2 : numpy.ndarray
        the linf error for tracer2

    l22 : numpy.ndarray
        the l2 error for tracer2

    linf3 : numpy.ndarray
        the linf error for tracer3

    l23 : numpy.ndarray
        the l2 error for tracer3
    """
    mSize = 8.0
    mWidth = mSize / 4
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    o1ref = 5 * np.array(dlambda)
    o2ref = 50 * np.square(dlambda)
    ax.loglog(
        dlambda,
        linf1,
        '+:',
        color=colors[0],
        markersize=mSize,
        markerfacecolor='none',
        markeredgewidth=mWidth,
        label="tracer1_linf")
    ax.loglog(
        dlambda,
        l21,
        '+-',
        color=colors[0],
        markersize=mSize,
        markerfacecolor='none',
        markeredgewidth=mWidth,
        label="tracer1_l2")
    ax.loglog(
        dlambda,
        linf2,
        's:',
        color=colors[1],
        markersize=mSize,
        markerfacecolor='none',
        markeredgewidth=mWidth,
        label="tracer2_linf")
    ax.loglog(
        dlambda,
        l22,
        's-',
        color=colors[1],
        markersize=mSize,
        markerfacecolor='none',
        markeredgewidth=mWidth,
        label="tracer2_l2")
    ax.loglog(
        dlambda,
        linf3,
        'v:',
        color=colors[2],
        markersize=mSize,
        markerfacecolor='none',
        markeredgewidth=mWidth,
        label="tracer3_linf")
    ax.loglog(
        dlambda,
        l23,
        'v-',
        color=colors[2],
        markersize=mSize,
        markerfacecolor='none',
        markeredgewidth=mWidth,
        label="tracer3_l2")
    ax.loglog(dlambda, o1ref, 'k--', label="1st ord.")
    ax.loglog(dlambda, o2ref, 'k-.', label="2nd ord.")
    ax.set_xticks(dlambda)
    ax.set_xticklabels(resvals)
    ax.tick_params(which='minor', labelbottom=False)
    ax.set(title=tcname, xlabel='QU res. val.', ylabel='rel. err.')
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')


def plot_filament(ax, tcname, resvals, filament):
    """
    Plot the "filament norm," as in figure 5 of LSPT2012.
    """
    tau = np.linspace(0, 1, 21)
    for i, r in enumerate(resvals):
        ax.plot(tau, filament[i], label="QU{}".format(r))
    ax.set(title=tcname, xlabel="tau", ylabel="rel. area")
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_ylim([0, 1.5])
    ax.grid()
    ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")


def plot_over_and_undershoot_errors(
        ax, tcname, resvals, u1, o1, u2, o2, u3, o3):
    """
    Plots over- and under-shoot error as a function of time.
    """
    if len(u1[0]) == 13:
        time = np.array(range(13))
        for i, r in enumerate(resvals):
            ax.plot(time, u1[i], label='QU{}_u1'.format(r))
            ax.plot(time, o1[i], label='QU{}_o1'.format(r))
            ax.plot(time, u2[i], label='QU{}_u2'.format(r))
            ax.plot(time, o2[i], label='QU{}_o2'.format(r))
            ax.plot(time, u3[i], label='QU{}_u3'.format(r))
            ax.plot(time, o3[i], label='QU{}_o3'.format(r))
        ax.set(
            title=tcname,
            xlabel='time (day)',
            ylabel='rel. over/undershoot')
