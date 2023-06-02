import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset

from compass.step import Step

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

        self.add_input_file(
            filename='output.nc',
            target='../forward/output.nc')
        self.add_input_file(
            filename='init.nc',
            target='../initial_state/initial_state.nc')
        self.add_output_file('plotTemp.png')

    def run(self):
        """
        Run this step of the test case
        """
        plt.figure(1, figsize=(11.0, 4.0))

        ncfileIC = Dataset('init.nc', 'r')
        ncfile = Dataset('output.nc', 'r')
        temp = ncfile.variables['density'][5, 0:500, :]
        temp = temp - 1000
        xCell = ncfileIC.variables['xCell'][0:500]
        zMid = ncfile.variables['zMid'][5, 0, :]
        L0 = 0.5
        a0 = 0.1
        x = xCell / L0
        z = zMid / a0
        plt.contour(x, z, temp.T, levels=[23, 23.25, 23.5, 23.75, 24,
                    24.25, 24.5, 24.75, 25, 25.25, 25.5, 25.75,
                    26, 26.25, 26.5, 26.75, 27], cmap='jet')
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 10, 20, 30, 40, 50])
        plt.yticks([0, -0.2, -0.4, -0.6, -0.8, -1], [0, -2, -4, -6, -8, -10])
        plt.xlabel('x, cm')
        plt.ylabel('z, cm')
        plt.colorbar(shrink=0.7)

        ncfileIC.close()
        ncfile.close()
        plt.savefig('plotTemp.png')
