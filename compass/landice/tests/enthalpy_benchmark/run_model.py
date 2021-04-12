import os
from netCDF4 import Dataset

from compass.model import run_model
from compass.step import Step


class RunModel(Step):
    """
    A step for performing forward MALI runs as part of dome test cases.

    Attributes
    ----------
    restart_filename : str, optional
        The name of a restart file to continue the run from
    """
    def __init__(self, test_case, name, restart_filename=None, subdir=None,
                 cores=1, min_cores=None, threads=1):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        name : str
            the name of the test case

        restart_filename : str, optional
            The name of a restart file to continue the run from

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        cores : int, optional
            the number of cores the step would ideally use.  If fewer cores
            are available on the system, the step will run on all available
            cores as long as this is not below ``min_cores``

        min_cores : int, optional
            the number of cores the step requires.  If the system has fewer
            than this number of cores, the step will fail

        threads : int, optional
            the number of threads the step will use
        """
        self.restart_filename = restart_filename
        if min_cores is None:
            min_cores = cores
        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         cores=cores, min_cores=min_cores, threads=threads)

        self.add_namelist_file('compass.landice.tests.enthalpy_benchmark',
                               'namelist.landice')
        self.add_streams_file('compass.landice.tests.enthalpy_benchmark',
                              'streams.landice')

        self.add_input_file(filename='landice_grid.nc',
                            target='../setup_mesh/landice_grid.nc')
        self.add_input_file(filename='graph.info',
                            target='../setup_mesh/graph.info')

        if restart_filename is not None:
            filename = os.path.basename(restart_filename)
            self.add_input_file(filename=filename, target=restart_filename)

        self.add_output_file(filename='output.nc')

    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies
        """
        self.add_model_as_input()

    def run(self):
        """
        Run this step of the test case
        """
        if self.restart_filename is not None:
            self._update_surface_air_temperature()

        run_model(self)

    def _update_surface_air_temperature(self):
        section = self.config['enthalpy_benchmark']
        phase = self.name
        # set the surface air temperature
        option = '{}_surface_air_temperature'.format(phase)
        surface_air_temperature = section.getfloat(option)
        filename = self.restart_filename
        with Dataset(filename, 'r+') as data:
            data.variables['surfaceAirTemperature'][0, :] = \
                surface_air_temperature
