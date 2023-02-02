from netCDF4 import Dataset
from compass.step import Step
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class Visualize(Step):
    """
    A step for visualizing a cross-section through the solitary wave
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
        self.add_output_file('plotTemp.png')

    def run(self):
        """
        Run this step of the test case
        """
        grids = ['nonhydro', 'hydro']
        nGrids = len(grids)
        plt.figure(1, figsize=(12.0, 6.0))

        config = self.config

        section = config['horizontal_grid']
        nx = section.getint('nx')
        section = config['visualize']
        maxLayerForPlot = section.getint('maxLayerForPlot')
        L0 = section.getint('L0')
        a0 = section.getint('a0')
        time = section.getint('plotTime')

        for j in range(nGrids):
            grid = grids[j] 
            ncfileIC = Dataset(f'init_{grid}.nc', 'r')
            ncfile = Dataset(f'output_{grid}.nc', 'r')
            temp = ncfile.variables['temperature'][time, 0:nx, :]
            xCell = ncfileIC.variables['xCell'][0:nx]
            zMid = ncfile.variables['zMid'][time, 0, :]
            x = xCell/L0
            z = zMid/a0
            z1 = z[0:maxLayerForPlot]
            temp1 = temp[:, 0:maxLayerForPlot]
            plt.ylabel('z/a0')
            plt.subplot(2, 1, j+1)
            plt.contour(x, z1, temp1.T)
            ncfileIC.close()
            ncfile.close()

        plt.xlabel('x/L0')
        plt.ylabel('z/a0')
        plt.savefig('plotTemp.png')
