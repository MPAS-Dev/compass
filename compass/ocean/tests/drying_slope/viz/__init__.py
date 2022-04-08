import os
import xarray
import numpy
import matplotlib.pyplot as plt
import pandas as pd

from compass.step import Step
from compass.ocean.tests.drying_slope.viz.plot import MoviePlotter, \
    TimeSeriesPlotter


class Viz(Step):
    """
    A step for visualizing drying slope results, as well as comparison with
    analytical solution and ROMS results.
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

        damping_coeffs = [0.0025, 0.01]
        times = ['0.50', '0.05', '0.40', '0.15', '0.30', '0.25']
        datatypes = ['analytical', 'ROMS']
        self.damping_coeffs = damping_coeffs
        self.times = times
        self.datatypes = datatypes

        for damping_coeff in damping_coeffs:
            self.add_input_file(filename='output_{}.nc'.format(damping_coeff),
                                target='../forward_{}/output.nc'.format(
                                       damping_coeff))
            for time in times:
                for datatype in datatypes:
                    filename = f'r{damping_coeff}d{time}-{datatype.lower()}.csv'
                    self.add_input_file(filename=f'comparison_data/{filename}', target=filename,
                                        database='drying_slope')

    def run(self):
        """
        Run this step of the test case
        """
        section = self.config['paths']
        datapath = section.get('ocean_database_root')
        section = self.config['drying_slope_viz']
        frames_per_second = section.getint('frames_per_second')
        movie_format = section.get('movie_format')

        tsPlotter = TimeSeriesPlotter()
        tsPlotter.plot_ssh_validation()
        tsPlotter.plot_ssh_time_series()

        mPlotter = MoviePlotter()
        mPlotter.plot_ssh_validation()
        mPlotter.images_to_movies(framesPerSecond=frames_per_second,
                                  extension=movie_format)
