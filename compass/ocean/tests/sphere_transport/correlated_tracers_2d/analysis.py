import numpy as np
from compass.step import Step
from ..process_output import *
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def init_triplot_axes(ax):
    lw = 0.4
    topline = Line2D([0.1, 1.0], [0.9, 0.9], color='k',
                     linestyle='-', linewidth=lw)
    botline = Line2D([0.1, 1.0], [0.9, 0.1], color='k',
                     linestyle='-', linewidth=lw)
    rightline = Line2D([1, 1], [0.1, 0.9], color='k',
                       linestyle='-', linewidth=lw)
    crvx = np.linspace(0.1, 1)
    crvy = -0.8 * np.square(crvx) + 0.9
    ticks = np.array(range(6)) / 5
    ax.plot(crvx, crvy, 'k-', linewidth=1.25 * lw)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.add_artist(topline)
    ax.add_artist(botline)
    ax.add_artist(rightline)
    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 1.1])
    ax.grid()


class Analysis(Step):
    """
    A step for visualizing the output from the correlatedTracers2D test case

    Attributes
    ----------
    resolutions : list of int
        The resolutions of the meshes that have been run
    """

    def __init__(self, test_case, resolutions):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.sphere_transport.correlatedTracers2D.CorrelatedTracers2D
            The test case this step belongs to

        resolutions : list of int
            The resolutions of the meshes that have been run
        """
        super().__init__(test_case=test_case, name='analysis')
        self.resolutions = resolutions
        self.tcdata = dict()

        for resolution in resolutions:
            self.add_input_file(
                filename='QU{}_namelist.ocean'.format(resolution),
                target='../QU{}/init/namelist.ocean'.format(resolution))
            self.add_input_file(
                filename='QU{}_init.nc'.format(resolution),
                target='../QU{}/init/initial_state.nc'.format(resolution))
            self.add_input_file(
                filename='QU{}_output.nc'.format(resolution),
                target='../QU{}/forward/output.nc'.format(resolution))
            self.add_output_file(
                'correlatedTracers2D_QU{}_sol.pdf'.format(resolution))
        self.add_output_file('correlatedTracers2D_triplots.png')

    def run(self):
        """
        Run this step of the test case
        """
        ###
        # Collect data
        ###
        for resolution in self.resolutions:
            ncd = Dataset('../QU{}/forward/output.nc'.format(resolution))
            self.tcdata[resolution] = {'dataset': ncd}
            self.tcdata[resolution]['appx_mesh_size'] = appx_mesh_size(ncd)
            self.tcdata[resolution]['err'] = compute_error_from_output_ncfile(
                ncd)
        print_data_as_csv('correlatedTracers2D', self.tcdata)

        ###
        # Plot solutions
        ###
        #   plt.rc('text', usetex=True) # .tex fails on Anvil
        plt.rc('font', family='sans-serif')
        plt.rc('ps', useafm=True)
        plt.rc('pdf', use14corefonts=True)
        for r in self.tcdata.keys():
            tcstr = 'correlatedTracers2D_QU{}'.format(r)
            fig = plt.figure(constrained_layout=True)
            plot_sol(fig, tcstr, self.tcdata[r]['dataset'])
            fig.savefig(tcstr + "_sol.pdf", bbox_inches='tight')
            plt.close(fig)

        ###
        # correlation analysis (aka "triangle plots")
        ###
        rvals = sorted(self.tcdata.keys())
        rvals.reverse()
        nrow = int(len(rvals) / 2)
        fig, axes = plt.subplots(nrow, 2, sharex=True, sharey=True)
        for i, r in enumerate(rvals):
            ax = axes[int(i / 2), i % 2]
            init_triplot_axes(ax)
            ax.set(title="QU{}".format(r))
            if i % 2 == 0:
                ax.set_ylabel("tracer3")
            if int(i / 2) == 2:
                ax.set_xlabel("tracer2")
            ds = self.tcdata[r]['dataset']
            ax.plot(ds.variables["tracer2"][6, :, 1],
                    ds.variables["tracer3"][6, :, 1], 'r.', markersize=1)
        fig.savefig("correlatedTracers2D_triplots.png")
