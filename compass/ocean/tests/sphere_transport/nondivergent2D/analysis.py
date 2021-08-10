import numpy as np
from compass.step import Step
from ..process_output import *
from netCDF4 import Dataset


class Analysis(Step):
    """
    A step for visualizing the output from the nondivergent2D test case

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
        test_case : compass.ocean.tests.sphere_transport.nondivergent2D.Nondivergent2D
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
            self.add_output_file('nondivergent2D_QU{}_sol.pdf'.format(resolution))

        self.add_output_file('nondivergent2D_convergence.pdf')

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
          tcstr = 'nondivergent2D_QU{}'.format(r)
          fig = plt.figure(constrained_layout=True)
          plot_sol(fig, tcstr, self.tcdata[r]['dataset'])
          fig.savefig(tcstr + "_sol.pdf", bbox_inches='tight')
          plt.close(fig)

        ###
        # convergence analysis
        ###
        dlambda, linf1, linf2, linf3, l21, l22, l23, fil, u1, o1, u2, o2, u3, o3 = make_convergence_arrays(self.tcdata)
        linfrate, l2rate = compute_convergence_rates(dlambda, linf1, l21)
        rvals = sorted(self.tcdata.keys())
        rvals.reverse()
        print_error_conv_table('nondivergent2D', rvals, dlambda, l21, l2rate, linf1, linfrate)

        fig, ax = plt.subplots()
        plot_convergence(ax, 'nondivergent2D', dlambda, rvals, linf1, l21, linf2, l22, linf3, l23)
        fig.savefig('nondivergent2D_convergence.pdf', bbox_inches='tight')
        plt.close(fig)

        ###
        # range and filament preservation
        ###
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(3,3)
        ax0 = fig.add_subplot(gs[0,:])
        plot_filament(ax0, 'nondivergent2D', rvals, fil)
        time = np.array(range(13))
        ctr = 0
        for i in range(1,3):
          for j in range(3):
            r = rvals[ctr]
            ax = fig.add_subplot(gs[i,j])
            ax.set(title="QU{}".format(r))
            ax.plot(time, u1[ctr], label='u1')
            ax.plot(time, o1[ctr], label='o1')
            ax.plot(time, u2[ctr], label='u2')
            ax.plot(time, o2[ctr], label='o2')
            ax.plot(time, u3[ctr], label='u3')
            ax.plot(time, o3[ctr], label='o3')
            ax.set_ylim((-0.5,0.5))
            if r == 60:
              ax.legend(bbox_to_anchor=(1,0.5), loc="center left")
            if j == 0:
              ax.set(ylabel="rel. range err.")
            ctr += 1

        fig.savefig('nondivergent2D_range_filament_err.pdf', bbox_inches='tight')
        plt.close(fig)

