import xarray
import numpy as np
from netCDF4 import Dataset
from compass.step import Step
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import transforms
matplotlib.use('Agg')


class Visualize(Step):
    """
    A step for visualizing a cross-section through the internal wave
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
        self.add_output_file('plotVertAndHor.png')

    def run(self):
        """
        Run this step of the test case
        """
        fig = plt.gcf()
        fig.set_size_inches(8.0, 10.0)

        initfile = Dataset(f'init_hydro.nc', 'r')
        ncfileH = Dataset(f'output_hydro.nc', 'r')
        ncfileNH = Dataset(f'output_nonhydro.nc', 'r')
        normalVelocityH = ncfileH.variables['normalVelocity']
        vertAleTransportTopH = ncfileH.variables['vertAleTransportTop']
        zMidH = ncfileH.variables['zMid']
        normalVelocityNH = ncfileNH.variables['normalVelocity']
        vertAleTransportTopNH = ncfileNH.variables['vertAleTransportTop']
        zMidNH = ncfileNH.variables['zMid']
        cellsOnEdge = initfile.variables['cellsOnEdge']
        edgesOnCell = initfile.variables['edgesOnCell']

        # horizontall velocity
        zMidEdge = 0.5*(zMidH[12, 31, :] + zMidH[12, 32, :])
        zMidEdge1 = zMidEdge/16
        print(np.shape(zMidEdge1))
        for i in range(0, 6):
            iEdge = edgesOnCell[31, i] - 1
            for j in range(0, 6):
                jEdge = edgesOnCell[32, j] - 1
                if (iEdge == jEdge):
                    midEdge = iEdge
        normalVelocity1 = normalVelocityH[12, midEdge, :]/ \
            max(normalVelocityH[12, midEdge, :])
        print(np.shape(normalVelocity1))
        zMidEdge = 0.5*(zMidNH[12, 31, :] + zMidNH[12, 32, :])
        zMidEdge2 = zMidEdge/16
        print(np.shape(zMidEdge2))
        for i in range(0, 6):
            iEdge = edgesOnCell[31, i] - 1
            for j in range(0, 6):
                jEdge = edgesOnCell[32, j] - 1
                if (iEdge == jEdge):
                    midEdge = iEdge
        normalVelocity2 = normalVelocityNH[12, midEdge, :]/ \
            max(normalVelocityNH[12, midEdge, :])
        print(np.shape(normalVelocity2))

        # vertical velocity
        zMid_origin1 = zMidH[12, 0, :]/16
        print(np.shape(zMid_origin1))
        vertAleTransportTop_origin1 = vertAleTransportTopH[12, 0, 0:100]/ \
            max(abs(vertAleTransportTopH[12, 0, 0:100]))
        print(np.shape(vertAleTransportTop_origin1))
        zMid_origin2 = zMidNH[12, 0, :]/16
        print(np.shape(zMid_origin2))
        vertAleTransportTop_origin2 = vertAleTransportTopNH[12, 0, 0:100]/ \
            max(abs(vertAleTransportTopNH[12, 0, 0:100]))
        print(np.shape(vertAleTransportTop_origin2))

        # plots
        plt.figure(figsize=(8.4, 4.2))
        plt.subplot(1, 2, 1)
        plt.plot(normalVelocity1, zMidEdge1, 'r')
        plt.plot(normalVelocity2, zMidEdge2, 'b')
        plt.xlabel('u/u_max')
        plt.ylabel('z/H')
        plt.yticks([0, -0.2, -0.4, -0.6, -0.8, -1])
        plt.title('Stratified Wave - hor profile')

        plt.subplot(1, 2, 2)
        plt.plot(vertAleTransportTop_origin1, zMid_origin1, 'r',
            label='H model')
        plt.plot(vertAleTransportTop_origin2, zMid_origin2, 'b',
            label='NH model')
        plt.xlim([-1.1, 1.1])
        plt.xlabel('w/w_max')
        plt.legend()
        plt.title('Stratified Wave - vert profile')

        ncfileH.close()
        ncfileNH.close()
        initfile.close()
        plt.savefig('plotVertAndHor.png')
