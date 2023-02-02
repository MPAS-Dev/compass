from netCDF4 import Dataset
from compass.step import Step
import matplotlib.pyplot as plt
import matplotlib

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

        config = self.config

        section = config['vertical_grid']
        nVertLevels = section.getint('vert_levels')
        section = config['visualize']
        time = section.getint('plotTime')
        cell1_midEdge = section.getint('cell1')
        cell2_midEdge = section.getint('cell2')
        firstCell = section.getint('firstCell')

        initfile = Dataset(f'init_hydro.nc', 'r')
        ncfileH = Dataset(f'output_hydro.nc', 'r')
        ncfileNH = Dataset(f'output_nonhydro.nc', 'r')
        normalVelocityH = ncfileH.variables['normalVelocity']
        vertAleTransportTopH = ncfileH.variables['vertAleTransportTop']
        zMidH = ncfileH.variables['zMid']
        normalVelocityNH = ncfileNH.variables['normalVelocity']
        vertAleTransportTopNH = ncfileNH.variables['vertAleTransportTop']
        zMidNH = ncfileNH.variables['zMid']
        edgesOnCell = initfile.variables['edgesOnCell']

        # horizontal velocity
        zMidEdge = 0.5 * (zMidH[time, cell1_midEdge, :] +
                          zMidH[time, cell2_midEdge, :])
        zMidEdge1 = zMidEdge / 16
        midEdge = None
        for i in range(0, 6):
            iEdge = edgesOnCell[cell1_midEdge, i] - 1
            for j in range(0, 6):
                jEdge = edgesOnCell[cell2_midEdge, j] - 1
                if iEdge == jEdge:
                    midEdge = iEdge
        if midEdge is None:
            raise ValueError('Could not find midEdge!')
        normalVelocity1 = \
            (normalVelocityH[time, midEdge, :] /
             max(normalVelocityH[time, midEdge, :]))
        zMidEdge = 0.5 * (zMidNH[time, cell1_midEdge, :] +
                          zMidNH[time, cell2_midEdge, :])
        zMidEdge2 = zMidEdge / 16
        for i in range(0, 6):
            iEdge = edgesOnCell[cell1_midEdge, i] - 1
            for j in range(0, 6):
                jEdge = edgesOnCell[cell2_midEdge, j] - 1
                if iEdge == jEdge:
                    midEdge = iEdge
        normalVelocity2 = (normalVelocityNH[time, midEdge, :] /
                           max(normalVelocityNH[time, midEdge, :]))

        # vertical velocity
        zMid_origin1 = zMidH[time, firstCell, :] / 16
        vertAleTransportTop_origin1 = \
            (vertAleTransportTopH[time, firstCell, 0:nVertLevels] /
             max(abs(vertAleTransportTopH[time, firstCell, 0:nVertLevels])))
        zMid_origin2 = zMidNH[time, firstCell, :] / 16
        vertAleTransportTop_origin2 = \
            (vertAleTransportTopNH[time, firstCell, 0:nVertLevels] /
             max(abs(vertAleTransportTopNH[time, firstCell, 0:nVertLevels])))

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
