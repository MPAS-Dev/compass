import os
import glob
import xarray
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
from scipy import spatial
from PIL import Image
from collections import OrderedDict

from compass.step import Step


class Viz(Step):
    """
    A step for visualizing dam break results, as well as comparison with
    experimental data and ROMS simulation output (Warner et al. 2013).

    Attributes
    ----------
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='viz')

        self.add_input_file(filename='output.nc',
                            target='../forward/output.nc')
        filename = 'stationCoords.csv'
        self.add_input_file(filename=filename, target=filename,
                            database='dam_break')

        filename = 'dam_break.png'
        self.add_input_file(filename=filename, target=filename,
                            database='dam_break')

        observed_inputs = ['Station0.csv', 'Station4.csv', 'Station-5A.csv',
                           'Station8A.csv', 'StationC.csv']
        roms_inputs = ['0-sim.csv', '4-sim.csv', '-5A-sim.csv', '8A-sim.csv',
                       'C-sim.csv']

        for filename in observed_inputs:
            self.add_input_file(filename=filename, target=filename,
                                database='dam_break')
        for filename in roms_inputs:
            self.add_input_file(filename=filename, target=filename,
                                database='dam_break')

        self.add_output_file(filename='dam_break_comparison.png')

    def run(self):
        """
        Run this step of the test case
        """
        # read output.nc
        data = xarray.open_dataset('output.nc')
        xCell = data.xCell.values
        yCell = data.yCell.values
        ssh = data.ssh.values
        ssh = ssh + 0.6
        dt = 0.3  # output interval in seconds
        nt, _ = numpy.shape(ssh)  # number of output times

        # read station coordinates
        StaData = pd.read_csv('stationCoords.csv', header=None)
        StaName = StaData.iloc[:, 0].values
        StaCoord = StaData.iloc[:, 1:]

        # Identify MPAS-O cells nearest observations
        # - coordinate shift from MPAS-O grid to dam break case
        xCell = 13 - xCell
        yCell = 13 - yCell

        # - find the nearest cell of each station
        matrix = numpy.array([xCell, yCell])
        tree = spatial.KDTree(list(zip(*matrix)))
        StaCell = tree.query(StaCoord)[1]

        # - cells representing station locations
        Station = OrderedDict(list(zip(StaName, StaCell)))

        ii = 0

        fig = plt.figure()
        for cell in Station:
            ii += 1
            ax = plt.subplot(3, 2, ii+1)

            # MPAS-O simulation results
            mpaso = plt.plot(numpy.arange(0, dt*nt, dt), ssh[:, Station[cell]],
                             color='#228B22', linewidth=2, alpha=0.6)

            # Measured data
            data = pd.read_csv('Station'+cell+'.csv', header=None)
            measured = plt.scatter(data[0], data[1], 4,
                                   marker='o', color='k')

            # ROMS simulation results (Warner et al., 2013)
            sim = pd.read_csv(cell+'-sim.csv', header=None)
            roms = plt.scatter(sim[0], sim[1], 4, marker='v', color='b')

            plt.xlim(0, 10)
            plt.xticks(numpy.arange(0, 11, 2))
            plt.ylim(0, 0.7)
            plt.yticks(numpy.arange(0, 0.7, 0.2))
            plt.text(3.5, 0.55, 'Station '+cell)

            if ii % 2 == 0:
                plt.ylabel('h (m)')
            if ii >= 4:
                plt.xlabel('time (s)')

            plt.tight_layout()
            plt.legend([mpaso[0], measured, roms],
                       ['MPAS-O', 'Measured', 'ROMS'],
                       fontsize='xx-small', frameon=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # station location map
        im = Image.open('dam_break.png')
        im2 = im.resize((650, 300))
        plt.subplot(3, 2, 1)
        plt.imshow(im2, interpolation='bicubic')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.locator_params(axis='x', nbins=3)
        plt.xticks([0, 325, 650], [4, 2, 0])
        plt.locator_params(axis='y', nbins=5)
        plt.yticks([0, 75, 150, 220, 300], reversed([2, 1.5, 1, 0.5, 0]))

        plt.savefig('./dam_break_comparison.png', dpi=600, bbox_inches='tight')
