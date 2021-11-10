import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from compass.step import Step


class ConvAnalysis(Step):
    """
    A step for visualizing and/or analyzing the output from a convergence test
    case

    Attributes
    ----------
    resolutions : list of int
        The resolutions of the meshes that have been run
    """
    def __init__(self, test_case, resolutions):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolutions : list of int
            The resolutions of the meshes that have been run
        """
        super().__init__(test_case=test_case, name='analysis')
        self.resolutions = resolutions

        # typically, the analysis will rely on the output from the forward
        # steps
        for resolution in resolutions:
            self.add_input_file(
                filename='{}km_output.nc'.format(resolution),
                target='../{}km/forward/output.nc'.format(resolution))
