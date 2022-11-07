import math
import numpy
import xarray
import matplotlib
import matplotlib.pyplot as plt
import cmocean

from compass.step import Step


class Analysis(Step):
    """
    A step for plotting the results of a series of TODO runs in the merry-go-
    round test group

    Attributes
    ----------
    resolution : str
        The resolution of the test case

    """
    def __init__(self, test_case, resolutions, name='analysis'):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : str
            The resolution of the test case

        """
        super().__init__(test_case=test_case, name=name)

        self.resolutions = resolutions

        self.add_output_file(filename='convergence_plot.png')

    def run(self):
        """
        Run this step of the test case
        """
        _plot(self.outputs[0], self.resolutions)


def _plot(filename, resolutions):
    """
    Plot section of the merry-go-round TODO

    Parameters
    ----------
    filename : str
        The output file name

    """
    # Note: ny does not currently get used
    plt.switch_backend('Agg')
    fig = plt.gcf()
    dt = [3, 6, 12] # nRefinement
    order2 = [0.4, 1.6, 6.4]
    operators = ['tracer1']
    nOperators = len(operators)

    L2 = numpy.zeros((len(resolutions)))

    for k in range(nOperators):
        for i, resolution in enumerate(resolutions):
            ds = xarray.open_dataset(f'../forward_{resolution}/output.nc')

            operator = operators[k]
            areas = ds.areaCell.values
            sol = ds[operator][1, :, 0].values
            ref = ds[operator][0, :, 0].values

            dif = abs(sol - ref)
            multDen = (ref**2)*areas
            multNum = (dif**2)*areas
            denL2 = numpy.sum(multDen[:])/numpy.sum(areas[:])
            numL2 = numpy.sum(multNum[:])/numpy.sum(areas[:])

            L2[i] = numpy.sqrt(numL2)/numpy.sqrt(denL2)

        order = math.log2(L2[0]/L2[1])
        print(order)
        order = math.log2(L2[1]/L2[2])
        print(order)
    
    for k in range(len(operators)):
        operator = operators[k]
        plt.loglog(dt, L2[:], '-x', label='rate')
    plt.loglog(dt, order2, label='slope=-2')
    plt.title('Convergence to the exact solution')
    plt.ylabel('l_2 error norm')
    plt.legend()
    plt.grid()
    plt.xticks(dt,dt)
    plt.xlabel('time steps (in min)')

    plt.savefig(filename)
