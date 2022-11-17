import math
import numpy
import xarray
import matplotlib
import matplotlib.pyplot as plt
import cmocean

from compass.step import Step


class Analysis(Step):
    """
    A step for plotting the convergence of the solution with resolution and
    time step in the merry-go-round test group

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

        resolutions : list of str
            The resolutions of the test case

        name: str
            The name of the step
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

    resolutions : list of str
        The resolutions of the test case
    """
    plt.switch_backend('Agg')
    fig = plt.gcf()
    dt = [3, 6, 12]
    order2 = [0.01, 0.04, 0.16]
    operators = ['tracer1']

    L2 = numpy.zeros((len(resolutions)))

    for k, operator in enumerate(operators):
        for i, resolution in enumerate(resolutions):
            ds = xarray.open_dataset(f'../forward_{resolution}/output.nc')

            areaCell = ds.areaCell.values
            final_field = ds[operator].isel(Time=1, nVertLevels=0).values
            initial_field = ds[operator].isel(Time=0, nVertLevels=0).values

            diff = abs(final_field - initial_field)
            multDen = (initial_field**2)*areaCell
            multNum = (diff**2)*areaCell
            denL2 = numpy.sum(multDen)/numpy.sum(areaCell)
            numL2 = numpy.sum(multNum)/numpy.sum(areaCell)

            L2[i] = numpy.sqrt(numL2)/numpy.sqrt(denL2)

        print(f'Order of convergence from dt 6 min to 3 min: ',
              f'{math.log2(L2[0]/L2[1])}')
        print(f'Order of convergence from dt 12 min to 6 min: ',
              f'{math.log2(L2[1]/L2[2])}')

        operator = operators[k]
        plt.loglog(dt, L2[:], '-x', label=f'Simulated {operator}')

    plt.loglog(dt, order2, 'k', label='Order 2 convergence')
    plt.title('Convergence to the exact solution')
    plt.ylabel('l_2 error norm')
    plt.legend()
    plt.grid()
    plt.xticks(dt, dt)
    plt.xlabel('time step (min)')

    plt.savefig(filename)
