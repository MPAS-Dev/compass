import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

from compass.step import Step

matplotlib.use('Agg')


class Visualize(Step):
    """
    A step for visualizing a cross-section through the fluid
    descend down the slope.

    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='visualize')

        for mode in ['nonhydro', 'hydro']:
            self.add_input_file(filename=f'output_{mode}.nc',
                                target=f'../{mode}/output.nc')
            self.add_input_file(filename=f'init_{mode}.nc',
                                target=f'../{mode}/init.nc')
        self.add_output_file('section_overflow.png')

    def run(self):
        """
        Run this step of the test case
        """
        modes = ['hydro', 'nonhydro']
        nModes = len(modes)
        plt.figure(1, figsize=(12.0, 6.0))

        config = self.config

        section = config['horizontal_grid']
        nx = section.getint('nx')
        dc = section.getfloat('dc')
        section = config['visualize']
        time = section.getint('plot_time')

        for j in range(nModes):
            mode = modes[j]
            ncfileIC = Dataset(f'init_{mode}.nc', 'r')
            ncfile = Dataset(f'output_{mode}.nc', 'r')
            temp = ncfile.variables['temperature']
            plt.subplot(2, 1, j + 1)
            plt.imshow(temp[time, 0:nx, :].T)
            plt.clim([10, 20])
            plt.jet()
            plt.colorbar()
            plt.xticks(np.arange(0., nx, 1000. / dc),
                       np.arange(0, nx * dc / 1000., 1))
            plt.yticks([0, 30, 60], [0, -100, -200])
            plt.xlabel('x, km')
            plt.ylabel('z, m')
            plt.title(f'temperature at 3hr - {mode}')
            ncfileIC.close()
            ncfile.close()

        plt.savefig('section_overflow.png')
