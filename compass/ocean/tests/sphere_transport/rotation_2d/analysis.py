import numpy as np
from compass.step import Step
from ..process_output import *
from netCDF4 import Dataset


class Analysis(Step):
    """
    A step for visualizing the output from the rotation_2d test case

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
        test_case : compass.ocean.tests.sphere_transport.rotation_2d.Rotation2D
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
            self.add_output_file('rotation_2d_QU{}_sol.pdf'.format(resolution))

        self.add_output_file('rotation_2d_convergence.pdf')

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
        print_data_as_csv('rotation_2d', self.tcdata)

        ###
        # Plot solutions
        ###
        #   plt.rc('text', usetex=True) # .tex fails on Anvil
        plt.rc('font', family='sans-serif')
        plt.rc('ps', useafm=True)
        plt.rc('pdf', use14corefonts=True)
        for r in self.tcdata.keys():
            tcstr = 'rotation_2d_QU{}'.format(r)
            fig = plt.figure(constrained_layout=True)
            plot_sol(fig, tcstr, self.tcdata[r]['dataset'])
            fig.savefig(tcstr + "_sol.pdf", bbox_inches='tight')

        ###
        # convergence analysis
        ###
        rvals = sorted(self.tcdata.keys())
        rvals.reverse()
        dlambda, linf1, linf2, linf3, l21, l22, l23, _, u1, o1, u2, o2, \
            u3, o3, mass1, mass2, mass3 = make_convergence_arrays(self.tcdata)
        linfrate, l2rate = compute_convergence_rates(dlambda, linf1, l21)
        linfrate, l2rate = compute_convergence_rates(dlambda, linf1, l21)
        print_error_conv_table(
            'rotation_2d',
            rvals,
            dlambda,
            l21,
            l2rate,
            linf1,
            linfrate)

        fig, ax = plt.subplots()
        plot_convergence(
            ax,
            'rotation_2d',
            dlambda,
            rvals,
            linf1,
            l21,
            linf2,
            l22,
            linf3,
            l23)
        fig.savefig('rotation_2d_convergence.pdf', bbox_inches='tight')
        plt.close()

        section = self.config['rotation_2d']

        all_above_thres = True
        error_message = ''
        for tracer in ['tracer1', 'tracer2', 'tracer3']:
            conv_thresh = section.getfloat(f'{tracer}_conv_thresh')
            l2_err = list()
            ncells = list()
            for resolution in self.resolutions:
                data = self.tcdata[resolution]
                l2_err.append(data['err'][tracer]['l2'])
                ncells.append(len(data['dataset'].dimensions["nCells"]))
            l2_err = np.array(l2_err)
            ncells = np.array(ncells)
            p = np.polyfit(np.log10(ncells), np.log10(l2_err), 1)

            # factor of 2 because nCells is like an inverse area, and we
            # want the convergence rate vs. cell size
            conv = abs(p[0]) * 2.0

            if conv < conv_thresh:
                all_above_thres = False
                error_message = \
                    f'{error_message}\n' \
                    f'            {tracer}: {conv:.2f} < {conv_thresh}'

        if not all_above_thres:
            raise ValueError('The following tracers have order of convergence '
                             '< min tolerance:' + error_message)
