import numpy as np
from compass.step import Step
from ..process_output import *
from netCDF4 import Dataset


class Analysis(Step):
    """
    A step for visualizing the output from the rotation2D test case

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
        test_case : compass.ocean.tests.sphere_transport.rotation2D.Rotation2D
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
            self.add_output_file('rotation2D_QU{}_sol.pdf'.format(resolution))

        self.add_output_file('rotation2D_convergence.pdf')

    def run(self):
        """
        Run this step of the test case
        """
        ###
        # Collect data
        ###
        for resolution in self.resolutions:
          ncd = Dataset('../QU{}/forward/output.nc'.format(resolution))
          self.tcdata[resolution] = {'dataset':ncd}
          self.tcdata[resolution]['appx_mesh_size'] = appx_mesh_size(ncd)
          self.tcdata[resolution]['err'] = compute_error_from_output_ncfile(ncd)

        ###
        # Plot solutions
        ###
        #   plt.rc('text', usetex=True) # .tex fails on Anvil
        plt.rc('font', family='sans-serif')
        plt.rc('ps', useafm=True)
        plt.rc('pdf', use14corefonts=True)
        for r in self.tcdata.keys():
          tcstr = 'rotation2D_QU{}'.format(r)
          fig = plt.figure(constrained_layout=True)
          plot_sol(fig, tcstr, self.tcdata[r]['dataset'])
          fig.savefig(tcstr + ".pdf", bbox_inches='tight')


        ###
        # convergence analysis
        ###
        rvals = sorted(self.tcdata.keys())
        rvals.reverse()
        dlambda = []
        linf1 = []
        linf2 = []
        linf3 = []
        l21 = []
        l22 = []
        l23 = []
        for r in rvals:
          dlambda.append(self.tcdata[r]['appx_mesh_size'])
          linf1.append(self.tcdata[r]['err']['tracer1']['linf'])
          linf2.append(self.tcdata[r]['err']['tracer2']['linf'])
          linf3.append(self.tcdata[r]['err']['tracer3']['linf'])
          l21.append(self.tcdata[r]['err']['tracer1']['l2'])
          l22.append(self.tcdata[r]['err']['tracer2']['l2'])
          l23.append(self.tcdata[r]['err']['tracer3']['l2'])
        linfrate, l2rate = compute_convergence_rates(dlambda, linf1, l21)
        print_error_conv_table('rotation2D', rvals, dlambda, l21, l2rate, linf1, linfrate)

        o1ref = 5*np.array(dlambda)
        o2ref = 50*np.square(dlambda)

        fig, ax = plt.subplots()
        mSize = 8.0
        mWidth = mSize/4
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        ax.loglog(dlambda, linf1, '+:', color=colors[0], markersize=mSize, markerfacecolor='none', markeredgewidth=mWidth, label="tracer1_linf")
        ax.loglog(dlambda, l21, '+-', color=colors[0], markersize=mSize, markerfacecolor='none', markeredgewidth=mWidth, label="tracer1_l2")
        ax.loglog(dlambda, linf2, 's:', color=colors[1], markersize=mSize, markerfacecolor='none', markeredgewidth=mWidth, label="tracer2_linf")
        ax.loglog(dlambda, l22, 's-', color=colors[1], markersize=mSize, markerfacecolor='none', markeredgewidth=mWidth, label="tracer2_l2")
        ax.loglog(dlambda, linf3, 'v:', color=colors[2], markersize=mSize, markerfacecolor='none', markeredgewidth=mWidth, label="tracer3_linf")
        ax.loglog(dlambda, l23, 'v-', color=colors[2], markersize=mSize, markerfacecolor='none', markeredgewidth=mWidth, label="tracer3_l2")
        ax.loglog(dlambda, o1ref, 'k--',label="1st ord.")
        ax.loglog(dlambda, o2ref, 'k-.', label="2nd ord.")
        ax.set_xticks(dlambda)
        ax.set_xticklabels(rvals)
        ax.tick_params(which='minor', labelbottom=False)
        ax.set(title='rotation2D', xlabel='QU res. val.', ylabel='rel. err.')
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        fig.savefig('rotation2D_convergence.pdf', bbox_inches='tight')
        plt.close()

