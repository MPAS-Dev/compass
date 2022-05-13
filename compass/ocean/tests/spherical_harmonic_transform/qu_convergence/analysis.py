import numpy as np
import matplotlib.pyplot as plt

from compass.step import Step


class Analysis(Step):
    """
    A step for visualizing the output from the spherical harmonic
    transformation test case

    Attributes
    ----------
    resolutions : list of int
        The resolutions of the meshes that have been run

    parallel_N : list of int
        The spherical harmonic orders used for the parallel algorithm

    serial_nLat : list of int
        The number of latitudes in the Gaussian meshes used for the
        serial algorithm
    """
    def __init__(self, test_case, resolutions, parallel_N, serial_nLat):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.spherical_harmonic_transform.qu_convergence.QuConvergence
            The test case this step belongs to

        resolutions : list of int
            The resolutions of the meshes that have been run

        parallel_N : list of int
            The spherical harmonic orders used for the parallel algorithm

        serial_nLat : list of int
            The number of latitudes in the Gaussian meshes used for the
            serial algorithm
        """
        super().__init__(test_case=test_case, name='analysis')
        self.resolutions = resolutions
        self.parallel_N = parallel_N
        self.serial_nLat = serial_nLat

        for resolution in resolutions:
            for N in parallel_N:
                self.add_input_file(
                    filename=f'QU{resolution}_parallel_N{N}',
                    target=f'../QU{resolution}/init/parallel/{N}/'
                           'log.ocean.0000.out')

            for nLat in serial_nLat:
                self.add_input_file(
                    filename=f'QU{resolution}_serial_nLat{nLat}',
                    target=f'../QU{resolution}/init/serial/{nLat}/'
                           'log.ocean.0000.out')

        self.add_output_file('convergence.png')

    def run(self):
        """
        Run this step of the test case
        """
        plt.switch_backend('Agg')
        resolutions = self.resolutions
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        dalpha = 0.9/float(len(resolutions)-1)
        alpha = 1.0
        for res in resolutions:
            serial_errors, serial_orders = self.get_errors(res, 'serial')
            parallel_errors, parallel_orders = self.get_errors(res, 'parallel')

            ax.loglog(serial_orders, serial_errors,
                      label=f'QU{res} serial', color='r', alpha=alpha)
            ax.loglog(parallel_orders, parallel_errors,
                      label=f'QU{res} parallel', color='b', alpha=alpha)

            alpha = alpha - dalpha

        ax.set_xlabel('Spherical harmonic order', fontsize=14)
        ax.set_ylabel('RMSE', fontsize=14)
        ax.legend(fontsize=14)
        plt.savefig('convergence.png', bbox_inches='tight', pad_inches=0.1)

        # if convergence_issue:
        #     raise ValueError(f'order of convergence '
        #                      f' {conv} < min tolerence {conv_thresh}')

    def get_errors(self, resolution, algorithm):
        """
        Get the errors for plotting and validation
        """
        if algorithm == 'parallel':
            N = self.parallel_N
            file_id = 'parallel_N'
        elif algorithm == 'serial':
            N = self.serial_nLat
            file_id = 'serial_nLat'

        errors = list()
        orders = list()
        for order in N:
            log_file = f'QU{resolution}_{file_id}{order}'
            errors.append(self.read_error(log_file))
            if algorithm == 'parallel':
                orders.append(order)
            elif algorithm == 'serial':
                orders.append((order-2)/2)

        errors = np.asarray(errors)
        orders = np.asarray(orders)

        return errors, orders

    def read_error(self, log_file):
        """
        Get error value for a given resolution

        Parameters
        ----------
        log_file : str
            The log file with the error value

        Returns
        -------
        error : float
            The root-mean-squared error
        """
        f = open(log_file, 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            if line.find('error RMS area weighted') >= 0:
                error = float(line.split()[-1])
                break

        return error
