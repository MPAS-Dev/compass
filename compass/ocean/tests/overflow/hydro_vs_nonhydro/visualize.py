import matplotlib
import matplotlib.pyplot as plt
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

        for grid in ['nonhydro', 'hydro']:
            self.add_input_file(filename=f'output_{grid}.nc',
                                target=f'../{grid}/output.nc')
            self.add_input_file(filename=f'init_{grid}.nc',
                                target=f'../{grid}/init.nc')
        self.add_output_file('section_overflow.png')

    def run(self):
        """
        Run this step of the test case
        """
        grids = ['hydro', 'nonhydro']
        nGrids = len(grids)
        plt.figure(1, figsize=(12.0, 6.0))

        config = self.config

        section = config['horizontal_grid']
        nx = section.getint('nx')
        section = config['visualize']
        time = section.getint('plot_time')

        for j in range(nGrids):
            grid = grids[j]
            ncfileIC = Dataset(f'init_{grid}.nc', 'r')
            ncfile = Dataset(f'output_{grid}.nc', 'r')
            temp = ncfile.variables['temperature']
            plt.subplot(2, 1, j + 1)
            plt.imshow(temp[time, 0:nx, :].T)
            plt.clim([10, 20])
            plt.jet()
            plt.colorbar()
            plt.xticks([0, 50, 100, 150, 200, 250, 300], [0, 1, 2, 3, 4, 5, 6])
            plt.yticks([0, 30, 60], [0, -100, -200])
            plt.xlabel('x, km')
            plt.ylabel('z, m')
            plt.title(f'temperature at 3hr - {grid}')
            ncfileIC.close()
            ncfile.close()

        plt.savefig('section_overflow.png')
